"""
Preprocess raw structure files into cached .pt files for GCQeq training.

This script converts ASE Atoms into torch-geometric graphs and precomputes
Ewald matrices so that training only needs to load the cache.

Usage:
    python preprocess.py \
        --train_data_path data/train.xyz \
        --valid_data_path data/valid.xyz \
        --output_dir ./data/preprocessed
"""

import argparse
import logging
import os

import torch
from ase.units import GPa
from tqdm import tqdm

from mattersim.datasets.utils.convertor import GraphConvertor
from mattersim.forcefield.m3gnet.modules.GCQeqSolver import precompute_ewald_data
from mattersim.utils.atoms_utils import AtomsAdaptor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_and_precompute(
    atoms_list,
    energies,
    forces,
    stresses,
    charges,
    fermi_values,
    cutoff=5.0,
    threebody_cutoff=4.0,
):
    """Convert atoms to graphs and attach precomputed Ewald data."""
    convertor = GraphConvertor("m3gnet", cutoff, True, threebody_cutoff)
    preprocessed = []

    for atoms, energy, force, stress, charge, fermi in tqdm(
        zip(atoms_list, energies, forces, stresses, charges, fermi_values),
        total=len(atoms_list),
        desc="Preprocessing graphs + Ewald",
    ):
        graph = convertor.convert(
            atoms.copy(), energy, force, stress, charge, fermi
        )
        if graph is None:
            continue

        ewald = precompute_ewald_data(graph.cell.squeeze(0), graph.atom_pos)
        graph.ewald_pair_i = ewald["pair_i"]
        graph.ewald_pair_j = ewald["pair_j"]
        graph.ewald_pair_shifts = ewald["pair_shifts"]
        graph.num_ewald_pairs = len(ewald["pair_i"])
        graph.ewald_k_vectors = ewald["k_vectors"]
        graph.ewald_k_lengths = ewald["k_lengths"]
        graph.num_ewald_kvecs = len(ewald["k_vectors"])
        graph.ewald_eta = ewald["eta"]
        graph.ewald_volume = ewald["volume"]

        preprocessed.append(graph)

    return preprocessed


def process_dataset(
    data_path,
    output_path,
    include_forces,
    include_stresses,
    include_charges,
    cutoff,
    threebody_cutoff,
):
    """Read a structure file, preprocess, and save to disk."""
    logger.info(f"Reading {data_path} ...")
    atoms_list = AtomsAdaptor.from_file(filename=data_path)

    energies = []
    forces_list = [] if include_forces else None
    stresses_list = [] if include_stresses else None
    charges_list = [] if include_charges else None
    fermi_values = []
    g_energies = []

    for atoms in atoms_list:
        energies.append(atoms.get_potential_energy())
        fermi_values.append(atoms.info["fermi"])
        g_energies.append(atoms.info["g_energy"])
        if include_forces:
            forces_list.append(atoms.get_forces())
        if include_stresses:
            stresses_list.append(atoms.get_stress(voigt=False) / GPa)
        if include_charges:
            charges_list.append(atoms.info["charge"])

    logger.info(f"  {len(atoms_list)} structures loaded")

    length = len(atoms_list)
    if forces_list is None:
        forces_list = [None] * length
    if stresses_list is None:
        stresses_list = [None] * length
    if charges_list is None:
        charges_list = [None] * length

    preprocessed = convert_and_precompute(
        atoms_list,
        g_energies,
        forces_list,
        stresses_list,
        charges_list,
        fermi_values,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
    )

    torch.save(preprocessed, output_path)
    logger.info(f"Saved {len(preprocessed)} preprocessed graphs to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess datasets for GCQeq training"
    )
    parser.add_argument("--train_data_path", type=str, required=None)
    parser.add_argument("--valid_data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./data")
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--threebody_cutoff", type=float, default=4.0)
    parser.add_argument(
        "--include_forces", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--include_stresses", action=argparse.BooleanOptionalAction, default=False
    )
    parser.add_argument(
        "--include_charges", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    if args.train_data_path is None and args.valid_data_path is None:
        logger.error("At least one of --train_data_path or --valid_data_path must be provided.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_data_path:
        train_out = os.path.join(args.output_dir, "train_preprocessed.pt")
        logger.info(f"Processing training data -> {train_out}")
        process_dataset(
            args.train_data_path,
            train_out,
            args.include_forces,
            args.include_stresses,
            args.include_charges,
            args.cutoff,
            args.threebody_cutoff,
        )

    if args.valid_data_path:
        valid_out = os.path.join(args.output_dir, "valid_preprocessed.pt")
        logger.info(f"Processing validation data -> {valid_out}")
        process_dataset(
            args.valid_data_path,
            valid_out,
            args.include_forces,
            args.include_stresses,
            args.include_charges,
            args.cutoff,
            args.threebody_cutoff,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()