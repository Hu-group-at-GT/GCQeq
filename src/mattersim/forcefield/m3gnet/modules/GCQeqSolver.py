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
    Precompute structure-dependent Ewald components (no sigma dependence).
    Reuses GCQeqSolver with point_charge=True to get the base matrix,
    and extracts filtered distances/indices for sigma correction at training time.

    Returns dict with:
        base_matrix       : (N, N) energy matrix assuming point charges
        filtered_distances: (K,)   pairwise distances within real-space cutoff
        pair_i, pair_j    : (K,)   atom indices for each filtered distance
    """
    dummy_sigmas = torch.zeros(pos.shape[0], device=pos.device)
    solver = GCQeqSolver(
        cell=cell, pos=pos, sigmas=dummy_sigmas,
        point_charge=True, acc_factor=acc_factor, eps=eps,
    )
    return {
        'base_matrix': solver.energy_matrix,
        'filtered_distances': solver.filtered_distances,
        'pair_i': solver.filtered_pair_i,
        'pair_j': solver.filtered_pair_j,
    }


def apply_sigma_correction(base_matrix, filtered_distances, pair_i, pair_j, sigmas):
    """
    Apply sigma-dependent Gaussian charge correction to a precomputed base matrix.

    Only two cheap operations:
      - Real space: subtract erfc(d / (sqrt(2) * gamma_ij)) / d for each filtered pair
      - Self energy: add 1 / (sqrt(pi) * sigma_i) on the diagonal

    Parameters
    ----------
    base_matrix         : (N, N) precomputed point-charge Ewald matrix (no grad)
    filtered_distances  : (K,)   distances within cutoff (no grad)
    pair_i, pair_j      : (K,)   atom index pairs (no grad)
    sigmas              : (N,)   learned Gaussian widths (carries gradient)
    """
    num_atoms = base_matrix.shape[0]

    # Real space correction: -erfc(d / (sqrt(2) * gamma_ij)) / d
    gammas = torch.sqrt(sigmas[pair_i] ** 2 + sigmas[pair_j] ** 2)
    corrections = -torch.erfc(filtered_distances / (math.sqrt(2) * gammas)) / filtered_distances

    correction_matrix = torch.zeros(
        num_atoms, num_atoms, device=sigmas.device, dtype=sigmas.dtype
    )
    correction_matrix = correction_matrix.index_put(
        (pair_i, pair_j), COULOMB_FACTOR * corrections, accumulate=True
    )

    # Self energy correction: +1 / (sqrt(pi) * sigma_i) on diagonal
    self_correction = COULOMB_FACTOR / (math.sqrt(math.pi) * sigmas.flatten())

    return base_matrix + correction_matrix + torch.diag(self_correction)


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