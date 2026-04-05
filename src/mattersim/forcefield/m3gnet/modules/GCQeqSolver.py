import torch
import math
from scipy import constants

COULOMB_FACTOR = 1e10 * constants.e / (4.0 * math.pi * constants.epsilon_0)

class GCQeqSolver:
    """Full Ewald matrix computation."""

    def __init__(
        self,
        cell: torch.Tensor,
        pos: torch.Tensor,
        sigmas: torch.Tensor,
        eta: float = None,
        cutoff_real: float = None,
        cutoff_recip: float = None,
        acc_factor: float = 12.0,
        point_charge: bool = False,
        eps: float = 1e-8,
    ):
        self.cell = cell
        self.volume = torch.abs(torch.det(self.cell))
        self.pos = pos
        self.sigmas = sigmas

        self.w = 1 / 2**0.5
        self.accf = math.sqrt(math.log(10**acc_factor))
        self.eta = (
            torch.tensor(eta)
            if eta is not None
            else (len(self.pos) * self.w / (self.volume**2)) ** (1 / 3) * math.pi
        )
        self.sqrt_eta = math.sqrt(self.eta)

        self.point_charge = point_charge
        self.eps = eps

        self.cutoff_real = cutoff_real or self.accf / self.sqrt_eta
        self.cutoff_recip = cutoff_recip or 2 * self.sqrt_eta * self.accf

        # precompute energy matrix
        e_real_matrix = self._calc_real_energy_matrix()
        e_recip_matrix = self._calc_reciprocal_energy_matrix()
        e_self_matrix = self._calc_self_energy_matrix()
        e_dipole_correction_matrix = self._calc_dipole_correction_matrix()

        self.energy_matrix = e_real_matrix + e_recip_matrix + e_self_matrix + e_dipole_correction_matrix

    @property
    def num_atoms(self):
        return self.pos.shape[0]

    def _calc_dipole_correction_matrix(self):
        """
        Compute dipole and constant-background charge correction matrix
        DOI: 10.1063/1.3216473 "Simulations of non-neutral slab systems with
        long-range electrostatic interactions in two-dimensional periodic boundary conditions"
        """
        prefac = (4.0 * math.pi / self.volume)

        z = self.pos[:, 2] # (N,)
        term1 = z.unsqueeze(1) * z.unsqueeze(0) # (N, 1) x (1, N)

        z_squared = z**2
        term2 = 0.5 * (z_squared.unsqueeze(1) + z_squared.unsqueeze(0))

        Lz = torch.norm(self.cell[2], dim=-1)
        ones = torch.ones(self.num_atoms, device=self.pos.device)
        term3 = (Lz**2 / 12.0) * ones.unsqueeze(1) * ones.unsqueeze(0)

        dipole_correction = prefac * (term1 - term2 - term3)

        return COULOMB_FACTOR * dipole_correction

    def _calc_real_energy_matrix(self):
        """
        Calculate real-space-part energy in atomic unit.
        Also stores filtered distances and atom indices on self for precomputation.
        """
        shifts = get_shifts_within_cutoff(self.cell, self.cutoff_real)
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        disps = disps_ij[None, :, :, :] + torch.matmul(shifts.to(self.cell.dtype), self.cell)[:, None, None, :]
        distances_all = torch.linalg.norm(disps, dim=-1)

        within_cutoff = (distances_all > self.eps) & (distances_all < self.cutoff_real)
        distances = distances_all[within_cutoff]

        # Store for precomputation use
        _, self.filtered_pair_i, self.filtered_pair_j = torch.where(within_cutoff)
        self.filtered_distances = distances

        e_real_matrix_aug = torch.zeros_like(distances_all)
        e_real_matrix_aug[within_cutoff] = torch.erfc(self.sqrt_eta * distances)
        if not self.point_charge:
            gammas_all = torch.sqrt(
                torch.square(self.sigmas.unsqueeze(1)) + torch.square(self.sigmas.unsqueeze(0))
            )
            gammas = torch.broadcast_to(gammas_all, distances_all.shape)[within_cutoff]
            e_real_matrix_aug[within_cutoff] -= torch.erfc(distances / (math.sqrt(2) * gammas))
        e_real_matrix_aug[within_cutoff] /= distances
        e_real_matrix = COULOMB_FACTOR * torch.sum(e_real_matrix_aug, dim=0)
        return e_real_matrix

    def _calc_reciprocal_energy_matrix(self):
        recip = get_reciprocal_vectors(self.cell)
        shifts = get_shifts_within_cutoff(recip, self.cutoff_recip)
        ks_all = torch.matmul(shifts.to(recip), recip)
        length_all = torch.linalg.norm(ks_all, dim=-1)

        within_cutoff = (length_all > self.eps) & (length_all < self.cutoff_recip)
        ks = ks_all[within_cutoff]
        length = length_all[within_cutoff]
        disps_ij = self.pos[None, :, :] - self.pos[:, None, :]
        phases = torch.sum(ks[:, None, None, :] * disps_ij[None, :, :, :], dim=-1)

        e_recip_matrix_aug = (
            torch.cos(phases)
            * torch.exp(-torch.square(length[:, None, None]) / (4 * self.eta))
            / torch.square(length[:, None, None])
        )
        e_recip_matrix = (
            COULOMB_FACTOR
            * 4.0
            * math.pi
            / self.volume
            * torch.sum(e_recip_matrix_aug, dim=0)
        )
        return e_recip_matrix

    def _calc_self_energy_matrix(self):
        device = self.pos.device
        diag = -2 * math.sqrt(self.eta / math.pi) * torch.ones(self.num_atoms, device=device)
        if not self.point_charge:
            diag += 1.0 / (math.sqrt(math.pi) * self.sigmas.flatten())
        e_self_matrix = COULOMB_FACTOR * torch.diag(diag)
        return e_self_matrix


# ---------------------------------------------------------------------------
# Precomputation for training
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_ewald_data(cell, pos, acc_factor=12.0, eps=1e-8):
    """
    Precompute structure-dependent Ewald neighbor lists and k-vectors.

    Only caches structural information (pair indices, shift vectors,
    reciprocal-space vectors, Ewald parameters).  The actual energy matrix
    is reconstructed at training time from live ``atom_pos`` via
    :func:`recompute_energy_matrix` so that autograd can differentiate
    through positions for correct force computation.

    Returns dict with:
        pair_i, pair_j  : (K,)   atom indices for real-space pairs
        pair_shifts     : (K, 3) fractional lattice shifts for each pair
        k_vectors       : (M, 3) reciprocal-space vectors within cutoff
        k_lengths       : (M,)   norms of k_vectors
        eta             : ()     Ewald splitting parameter
        volume          : ()     cell volume
    """
    num_atoms = pos.shape[0]
    volume = torch.abs(torch.det(cell))
    w = 1 / 2**0.5
    accf = math.sqrt(math.log(10**acc_factor))
    eta = (num_atoms * w / (volume**2)) ** (1 / 3) * math.pi
    sqrt_eta = math.sqrt(eta)
    cutoff_real = accf / sqrt_eta
    cutoff_recip = 2 * sqrt_eta * accf

    # Real space: find pairs within cutoff and record their shift vectors
    shifts = get_shifts_within_cutoff(cell, cutoff_real)
    disps_ij = pos[None, :, :] - pos[:, None, :]
    disps = disps_ij[None, :, :, :] + torch.matmul(
        shifts.to(cell.dtype), cell
    )[:, None, None, :]
    distances_all = torch.linalg.norm(disps, dim=-1)
    within_cutoff = (distances_all > eps) & (distances_all < cutoff_real)
    shift_idx, pair_i, pair_j = torch.where(within_cutoff)
    pair_shifts = shifts[shift_idx]  # fractional lattice shift per pair

    # Reciprocal space: find k-vectors within cutoff
    recip = get_reciprocal_vectors(cell)
    recip_shifts = get_shifts_within_cutoff(recip, cutoff_recip)
    ks_all = torch.matmul(recip_shifts.to(recip.dtype), recip)
    length_all = torch.linalg.norm(ks_all, dim=-1)
    k_mask = (length_all > eps) & (length_all < cutoff_recip)
    k_vectors = ks_all[k_mask]
    k_lengths = length_all[k_mask]

    return {
        'pair_i': pair_i,
        'pair_j': pair_j,
        'pair_shifts': pair_shifts,
        'k_vectors': k_vectors,
        'k_lengths': k_lengths,
        'eta': torch.tensor(eta, dtype=cell.dtype),
        'volume': volume,
    }


def recompute_energy_matrix(pos, cell, sigmas, pair_i, pair_j, pair_shifts,
                            k_vectors, k_lengths, eta, volume):
    """
    Reconstruct the full Ewald energy matrix with live positions in the
    autograd graph, enabling correct force computation via backprop.

    Cached structural data (pair indices, shifts, k-vectors) avoids the
    expensive neighbor search, while positions are recomputed so that
    ``torch.autograd.grad(energy, atom_pos)`` captures Coulomb forces.

    Parameters
    ----------
    pos          : (N, 3) atom positions (requires_grad=True)
    cell         : (3, 3) unit cell
    sigmas       : (N,)   learned Gaussian widths
    pair_i/j     : (K,)   atom index pairs from cache
    pair_shifts  : (K, 3) fractional lattice shifts from cache
    k_vectors    : (M, 3) reciprocal-space vectors from cache
    k_lengths    : (M,)   norms of k_vectors from cache
    eta          : float  Ewald splitting parameter
    volume       : float  cell volume
    """
    num_atoms = pos.shape[0]
    eta_val = float(eta)
    vol_val = float(volume)
    sqrt_eta = math.sqrt(eta_val)

    # --- Real space (autograd through pos) ---
    disps = pos[pair_j] - pos[pair_i] + pair_shifts @ cell  # (K, 3)
    distances = torch.linalg.norm(disps, dim=-1)  # (K,)

    # Point-charge erfc term
    real_vals = torch.erfc(sqrt_eta * distances) / distances
    # Sigma correction: subtract Gaussian-smeared part
    gammas = torch.sqrt(sigmas[pair_i] ** 2 + sigmas[pair_j] ** 2)
    real_vals = real_vals - torch.erfc(
        distances / (math.sqrt(2) * gammas)
    ) / distances

    real_matrix = torch.zeros(
        num_atoms, num_atoms, device=pos.device, dtype=pos.dtype
    )
    real_matrix = real_matrix.index_put(
        (pair_i, pair_j), COULOMB_FACTOR * real_vals, accumulate=True
    )

    # --- Reciprocal space (autograd through pos) ---
    disps_ij = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
    phases = torch.einsum('kd,ijd->kij', k_vectors, disps_ij)  # (M, N, N)
    recip_weights = (
        torch.exp(-k_lengths ** 2 / (4 * eta_val)) / k_lengths ** 2
    )  # (M,)
    recip_matrix = (
        COULOMB_FACTOR * 4.0 * math.pi / vol_val
        * torch.einsum('k,kij->ij', recip_weights, torch.cos(phases))
    )

    # --- Self energy ---
    diag = -2 * math.sqrt(eta_val / math.pi) * torch.ones(
        num_atoms, device=pos.device, dtype=pos.dtype
    )
    diag = diag + 1.0 / (math.sqrt(math.pi) * sigmas.flatten())
    self_matrix = COULOMB_FACTOR * torch.diag(diag)

    # --- Dipole correction (autograd through pos z-coordinates) ---
    z = pos[:, 2]
    prefac = 4.0 * math.pi / vol_val
    term1 = z.unsqueeze(1) * z.unsqueeze(0)
    z_sq = z ** 2
    term2 = 0.5 * (z_sq.unsqueeze(1) + z_sq.unsqueeze(0))
    Lz = torch.norm(cell[2], dim=-1)
    ones = torch.ones(num_atoms, device=pos.device, dtype=pos.dtype)
    term3 = (Lz ** 2 / 12.0) * ones.unsqueeze(1) * ones.unsqueeze(0)
    dipole_matrix = COULOMB_FACTOR * prefac * (term1 - term2 - term3)

    return real_matrix + recip_matrix + self_matrix + dipole_matrix


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def get_reciprocal_vectors(cell):
    """
    Return reciprocal vectors of `cell`.
    Let the returned matrix be recip, dot(cell[i, :], recip[j, :]) = 2 * pi * (i == j)
    """
    recip = 2 * math.pi * torch.transpose(torch.linalg.inv(cell), 0, 1)
    return recip


def get_shifts_within_cutoff(cell, cutoff):
    """
    Return all shifts required to search for atoms within cutoff
    """
    device = cell.device

    # projected length for three planes
    proj = torch.zeros(3, device=device)
    nx = torch.linalg.cross(cell[1], cell[2])
    ny = torch.linalg.cross(cell[2], cell[0])
    nz = torch.linalg.cross(cell[0], cell[1])
    proj[0] = torch.dot(cell[0], nx / torch.linalg.norm(nx))
    proj[1] = torch.dot(cell[1], ny / torch.linalg.norm(ny))
    proj[2] = torch.dot(cell[2], nz / torch.linalg.norm(nz))

    shift = torch.ceil(cutoff / torch.abs(proj))
    grid = torch.cartesian_prod(
        torch.arange(-float(shift[0]), float(shift[0]) + 1, device=device),
        torch.arange(-float(shift[1]), float(shift[1]) + 1, device=device),
        torch.arange(-float(shift[2]), float(shift[2]) + 1, device=device),
    )

    return grid