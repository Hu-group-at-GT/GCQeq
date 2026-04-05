# -*- coding: utf-8 -*-
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_runstats.scatter import scatter

from mattersim.jit_compile_tools.jit import compile_mode

from .modules import (  # noqa: F501
    MLP,
    GatedMLP,
    MainBlock,
    SmoothBesselBasis,
    SphericalBasisLayer,
)
from .scaling import AtomScaling

from .modules.GCQeqSolver import GCQeqSolver, recompute_energy_matrix
from .modules.init_elemental_properties import get_sigma


@compile_mode("script")
class M3Gnet(nn.Module):
    """
    M3Gnet
    """

    def __init__(
        self,
        num_blocks: int = 4,
        units: int = 128,
        max_l: int = 4,
        max_n: int = 4,
        cutoff: float = 5.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        max_z: int = 94,
        threebody_cutoff: float = 4.0,
        **kwargs,
    ):
        super().__init__()
        self.rbf = SmoothBesselBasis(r_max=cutoff, max_n=max_n)
        self.sbf = SphericalBasisLayer(max_n=max_n, max_l=max_l, cutoff=cutoff)
        self.edge_encoder = MLP(
            in_dim=max_n, out_dims=[units], activation="swish", use_bias=False
        )
        module_list = [
            MainBlock(max_n, max_l, cutoff, units, max_n, threebody_cutoff)
            for i in range(num_blocks)
        ]

        self.graph_conv = nn.ModuleList(module_list)

        self.MLP1 = GatedMLP(
            in_dim=units+1,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.MLP2 = GatedMLP(
            in_dim=units+1, 
            out_dims=[units, units, 1],
            activation=["swish", "swish", "softplus"]
        )
        self.MLP3 = GatedMLP(
            in_dim=units+1,
            out_dims=[units, units, 1],
            activation=["swish", "swish", None],
        )
        self.MLP4 = GatedMLP(
            in_dim=units+1,
            out_dims=[units, units, 1],
            activation=["swish", "swish", "softplus"],
        )
        self.apply(self.init_weights)
        self.atom_embedding = MLP(
            in_dim=max_z + 1, out_dims=[units], activation=None, use_bias=False
        )
        self.atom_embedding.apply(self.init_weights_uniform)
        self.normalizer = AtomScaling(verbose=False, max_z=max_z)
        self.max_z = max_z
        self.device = device
        self.model_args = {
            "num_blocks": num_blocks,
            "units": units,
            "max_l": max_l,
            "max_n": max_n,
            "cutoff": cutoff,
            "max_z": max_z,
            "threebody_cutoff": threebody_cutoff,
        }

        self.register_buffer("sigma", get_sigma(self.max_z, device=device))

    def forward(
        self,
        input: Dict[str, torch.Tensor],
        dataset_idx: int = -1,
    ) -> torch.Tensor:
        # Exact data from input_dictionary
        pos = input["atom_pos"]
        cell = input["cell"]
        pbc_offsets = input["pbc_offsets"].float()
        atom_attr = input["atom_attr"]
        edge_index = input["edge_index"].long()
        three_body_indices = input["three_body_indices"].long()
        num_three_body = input["num_three_body"]
        num_bonds = input["num_bonds"]
        num_triple_ij = input["num_triple_ij"]
        num_atoms = input["num_atoms"]
        num_graphs = input["num_graphs"]
        fermi = input["fermi"]
        batch = input["batch"]

        # -------------------------------------------------------------#
        cumsum = torch.cumsum(num_bonds, dim=0) - num_bonds
        index_bias = torch.repeat_interleave(  # noqa: F501
            cumsum, num_three_body, dim=0
        ).unsqueeze(-1)
        three_body_indices = three_body_indices + index_bias

        # === Refer to the implementation of M3GNet,        ===
        # === we should re-compute the following attributes ===
        # edge_length, edge_vector(optional), triple_edge_length, theta_jik
        atoms_batch = torch.repeat_interleave(repeats=num_atoms)
        edge_batch = atoms_batch[edge_index[0]]
        edge_vector = pos[edge_index[0]] - (
            pos[edge_index[1]]
            + torch.einsum("bi, bij->bj", pbc_offsets, cell[edge_batch])
        )
        edge_length = torch.linalg.norm(edge_vector, dim=1)
        vij = edge_vector[three_body_indices[:, 0].clone()]
        vik = edge_vector[three_body_indices[:, 1].clone()]
        rij = edge_length[three_body_indices[:, 0].clone()]
        rik = edge_length[three_body_indices[:, 1].clone()]
        cos_jik = torch.sum(vij * vik, dim=1) / (rij * rik)
        # eps = 1e-7 avoid nan in torch.acos function
        cos_jik = torch.clamp(cos_jik, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        triple_edge_length = rik.view(-1)
        edge_length = edge_length.unsqueeze(-1)
        atomic_numbers = atom_attr.squeeze(1).long()

        # featurize
        atom_attr = self.atom_embedding(self.one_hot_atoms(atomic_numbers))
        edge_attr = self.rbf(edge_length.view(-1))
        edge_attr_zero = edge_attr  # e_ij^0
        edge_attr = self.edge_encoder(edge_attr)
        three_basis = self.sbf(triple_edge_length, torch.acos(cos_jik))

        sigma = self.sigma[atomic_numbers]

        # Main Loop
        for idx, conv in enumerate(self.graph_conv):
            atom_attr, edge_attr = conv(
                atom_attr,
                edge_attr,
                edge_attr_zero,
                edge_index,
                three_basis,
                three_body_indices,
                edge_length,
                num_bonds,
                num_triple_ij,
                num_atoms,
            )

        fermi_per_atom = fermi[batch].unsqueeze(-1)  # shape: [num_atoms, 1]
        aev_with_fermi = torch.cat([atom_attr, fermi_per_atom], dim=1)

        chi = self.MLP1(aev_with_fermi).view(-1)  # [batch_size*num_atoms]
        J = self.MLP2(aev_with_fermi).view(-1)  # [batch_size*num_atoms]
        sigma_factor = self.MLP4(aev_with_fermi).view(-1)  # [batch_size*num_atoms]

        # Build and solve Ewald system per graph (serial)
        has_precomputed = "ewald_pair_shifts" in input
        if has_precomputed:
            ewald_pi = input["ewald_pair_i"]
            ewald_pj = input["ewald_pair_j"]
            ewald_shifts = input["ewald_pair_shifts"]
            num_ewald_pairs = input["num_ewald_pairs"]
            ewald_kvecs = input["ewald_k_vectors"]
            ewald_klens = input["ewald_k_lengths"]
            num_ewald_kvecs = input["num_ewald_kvecs"]
            ewald_eta = input["ewald_eta"]
            ewald_volume = input["ewald_volume"]
            pair_offset = 0
            kvec_offset = 0

        Q_list = []
        Q_tot_list = []
        omega_list = []
        E_coulomb_list = []

        atom_offset = 0
        for i in range(num_graphs):
            n = num_atoms[i].item()

            chi_i = chi[atom_offset:atom_offset + n]
            J_i = J[atom_offset:atom_offset + n]
            sigma_i = sigma[atom_offset:atom_offset + n] * sigma_factor[atom_offset:atom_offset + n]
            fermi_i = fermi[i]

            if has_precomputed:
                # Training path: recompute energy matrix from cached structural
                # data with live pos in the autograd graph for correct forces
                pos_i = pos[atom_offset:atom_offset + n]
                cell_i = cell[i]
                k_i = num_ewald_pairs[i].item()
                pi_i = ewald_pi[pair_offset:pair_offset + k_i]
                pj_i = ewald_pj[pair_offset:pair_offset + k_i]
                shifts_i = ewald_shifts[pair_offset:pair_offset + k_i]
                m_i = num_ewald_kvecs[i].item()
                kvecs_i = ewald_kvecs[kvec_offset:kvec_offset + m_i]
                klens_i = ewald_klens[kvec_offset:kvec_offset + m_i]
                energy_matrix_i = recompute_energy_matrix(
                    pos_i, cell_i, sigma_i,
                    pi_i, pj_i, shifts_i,
                    kvecs_i, klens_i,
                    ewald_eta[i], ewald_volume[i],
                )
                pair_offset += k_i
                kvec_offset += m_i
            else:
                # Inference path: full Ewald computation
                pos_i = pos[atom_offset:atom_offset + n]
                cell_i = cell[i]
                gcqeq = GCQeqSolver(
                    cell=cell_i, pos=pos_i, sigmas=sigma_i, point_charge=False
                )
                energy_matrix_i = gcqeq.energy_matrix

            A_i = energy_matrix_i + torch.diag(J_i)
            RHS_i = chi_i + fermi_i

            Q_i = -torch.linalg.solve(A_i, RHS_i)

            omega_i = 0.5 * Q_i @ (A_i @ Q_i) + RHS_i @ Q_i
            e_coulomb_i = 0.5 * torch.sum(energy_matrix_i * Q_i[:, None] * Q_i[None, :])

            Q_list.append(Q_i)
            Q_tot_list.append(Q_i.sum())
            omega_list.append(omega_i)
            E_coulomb_list.append(e_coulomb_i)

            atom_offset += n

        Q_tot = torch.stack(Q_tot_list)
        omega = torch.stack(omega_list)
        E_coulomb = torch.stack(E_coulomb_list)

        Q_star = torch.cat(Q_list, dim=0).unsqueeze(-1)
        aev_with_Q = torch.cat([atom_attr, Q_star], dim=1)

        energies_i = self.MLP3(aev_with_Q).view(-1) # [batch_size*num_atoms]
        energies_i = self.normalizer(energies_i, atomic_numbers)

        energies = scatter(energies_i, batch, dim=0, dim_size=num_graphs) + omega

        return energies, Q_tot, Q_star, chi, J, omega, E_coulomb, sigma_factor # [batch_size], [batch_size]

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    def init_weights_uniform(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, a=-0.05, b=0.05)

    @torch.jit.export
    def one_hot_atoms(self, species):
        # one_hots = []
        # for i in range(species.shape[0]):
        #     one_hots.append(
        #         F.one_hot(
        #             species[i],
        #             num_classes=self.max_z+1).float().to(species.device)
        #     )
        # return torch.cat(one_hots, dim=0)
        return F.one_hot(species, num_classes=self.max_z + 1).float()

    def print(self):
        from prettytable import PrettyTable

        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            params = parameter.numel()
            table.add_row([name, params])
            total_params += params
        print(table)
        print(f"Total Trainable Params: {total_params}")

    @torch.jit.export
    def set_normalizer(self, normalizer: AtomScaling):
        self.normalizer = normalizer

    def get_model_args(self):
        return self.model_args