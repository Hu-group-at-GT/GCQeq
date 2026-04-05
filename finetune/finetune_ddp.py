import argparse
import logging
import os
import pickle as pkl
from ase.units import GPa

import torch
import torch.distributed

from mattersim.utils.atoms_utils import AtomsAdaptor
from mattersim.datasets.utils.build import build_dataloader
from mattersim.forcefield.m3gnet.scaling import AtomScaling
from mattersim.forcefield.m3gnet.m3gnet import M3Gnet
from mattersim.forcefield.potential import Potential

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

local_rank = int(os.environ.get("LOCAL_RANK", 0))


def _read_extxyz(data_path, include_forces, include_stresses, include_charges):
    """Read an extxyz file and return atoms + label arrays."""
    atoms_list = AtomsAdaptor.from_file(filename=data_path)

    energies, forces, stresses, charges, fermi_values, g_energies = (
        [], [] if include_forces else None, [] if include_stresses else None,
        [] if include_charges else None, [], [],
    )
    for atoms in atoms_list:
        energies.append(atoms.get_potential_energy())
        fermi_values.append(atoms.info["fermi"])
        g_energies.append(atoms.info["g_energy"])
        if include_forces:
            forces.append(atoms.get_forces())
        if include_stresses:
            stresses.append(atoms.get_stress(voigt=False) / GPa)
        if include_charges:
            charges.append(atoms.info["charge"])

    return atoms_list, energies, forces, stresses, charges, fermi_values, g_energies


def main(args):
    is_distributed = args.is_distributed
    args_dict = vars(args)

    if is_distributed:
        if args.device == "cuda":
            torch.distributed.init_process_group(
                backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
            )
            torch.cuda.set_device(local_rank)
        else:
            torch.distributed.init_process_group(backend="gloo")
        torch.distributed.barrier()

    device = args.device
    use_cached_train = args.train_preprocessed_path is not None
    use_cached_valid = args.valid_preprocessed_path is not None

    # ------------------------------------------------------------------
    # Training dataloader
    # ------------------------------------------------------------------
    if use_cached_train:
        logger.info("Using preprocessed training cache: %s", args.train_preprocessed_path)
        dataloader = build_dataloader(
            preprocessed_path=args.train_preprocessed_path,
            shuffle=True,
            pin_memory=(device == "cuda"),
            **args_dict,
        )
    else:
        assert args.train_data_path is not None, (
            "Either --train_preprocessed_path or --train_data_path must be provided"
        )
        logger.info("Reading training data from extxyz: %s", args.train_data_path)
        atoms_train, energies, forces, stresses, charges, fermi_values, g_energies = (
            _read_extxyz(
                args.train_data_path,
                args.include_forces, args.include_stresses, args.include_charges,
            )
        )
        dataloader = build_dataloader(
            atoms_train, g_energies, charges, fermi_values, forces, stresses,
            shuffle=True, pin_memory=(device == "cuda"), **args_dict,
        )

    if args.re_normalize:
        if use_cached_train:
            assert args.train_data_path is not None, (
                "--re_normalize requires --train_data_path to compute scaling stats"
            )
            logger.info("Reading raw training data for re-normalization...")
            atoms_train, _, forces, *_ = _read_extxyz(
                args.train_data_path,
                args.include_forces, False, False,
            )
            g_energies = [a.info["g_energy"] for a in atoms_train]

        scale = AtomScaling(
            atoms=atoms_train,
            total_energy=g_energies,
            forces=forces,
            verbose=True,
            **args_dict,
        ).to(device)

    if use_cached_valid:
        logger.info("Using preprocessed validation cache: %s", args.valid_preprocessed_path)
        # val_args = {**args_dict, "is_distributed": False}
        val_dataloader = build_dataloader(
            preprocessed_path=args.valid_preprocessed_path,
            pin_memory=(device == "cuda"),
            **args_dict,
        )
    elif args.valid_data_path is not None:
        logger.info("Reading validation data from extxyz: %s", args.valid_data_path)
        atoms_val, _, forces_v, stresses_v, charges_v, fermi_v, g_energies_v = (
            _read_extxyz(
                args.valid_data_path,
                args.include_forces, args.include_stresses, args.include_charges,
            )
        )
        val_dataloader = build_dataloader(
            atoms_val, g_energies_v, charges_v, fermi_v, forces_v, stresses_v,
            pin_memory=(device == "cuda"), **args_dict,
        )
    else:
        val_dataloader = None

    potential = Potential.from_checkpoint(
        load_path="../pretrained_models/mattersim-v1.0.0-1M.pth",
        load_training_state=False,
        scheduler="ReduceLROnPlateau",
        **args_dict,
    )

    if args.re_normalize:
        potential.model.set_normalizer(scale)

    if is_distributed and args.device == "cuda":
        potential.model = torch.nn.parallel.DistributedDataParallel(
            potential.model, device_ids=[local_rank]
        )
        torch.distributed.barrier()

    potential.train_model(
        dataloader=dataloader,
        val_dataloader=val_dataloader,
        is_distributed=is_distributed,
        include_energy=True,
        include_forces=True,
        include_stresses=False,
        include_charges=True,
        epochs=201,
        save_checkpoint=True,
        early_stop_patience = 200,
        save_path="./results",
        force_loss_ratio = 0.2,
        charge_loss_ratio = 0.2
    )

    if not is_distributed or local_rank == 0:
        potential.save("./results/final_model_finetuned.pth")

    metrics = potential.test_model(val_dataloader, is_distributed=is_distributed)
    if not is_distributed or local_rank == 0:
        print("Final Metrics (loss, MAE_energy, MAE_force, MAE_stress):", metrics)

    if is_distributed:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adapt MatterSim")
    parser.add_argument("--train_data_path", type=str, default=None,
                        help="Path to raw extxyz (not needed if --train_preprocessed_path is set)")
    parser.add_argument(
        "--valid_data_path", type=str, default=None, help="valid data path"
    )
    parser.add_argument("--train_preprocessed_path", type=str, default=None,
                        help="Path to cached preprocessed training data (.pt)")
    parser.add_argument("--valid_preprocessed_path", type=str, default=None,
                        help="Path to cached preprocessed validation data (.pt)")

    parser.add_argument(
        "--save_path", type=str, default="./results", help="path to save the model"
    )
    parser.add_argument(
        "--save_checkpoint",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=10,
        help="save checkpoint every ckpt_interval epochs",
    )

    parser.add_argument(
        "--re_normalize",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="re-normalize the energy and forces according to the new data",
    )
    parser.add_argument("--scale_key", type=str, default="per_species_forces_rms")
    parser.add_argument(
        "--shift_key", type=str, default="per_species_energy_mean_linear_reg"
    )
    parser.add_argument("--init_scale", type=float, default=None)
    parser.add_argument("--init_shift", type=float, default=None)
    parser.add_argument(
        "--trainable_scale",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--trainable_shift",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    parser.add_argument("--device", type=str, default="cuda", help="device to use")
    parser.add_argument(
        "--is_distributed",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
        help="enable multi-GPU distributed training (launch with torchrun)",
    )

    # model parameters
    parser.add_argument("--cutoff", type=float, default=5.0, help="cutoff radius")
    parser.add_argument(
        "--threebody_cutoff",
        type=float,
        default=4.0,
        help="cutoff radius for three-body term, which should be smaller than cutoff (two-body)",  # noqa: E501
    )

    # training parameters
    parser.add_argument("--epochs", type=int, default=1000, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument(
        "--step_size",
        type=int,
        default=10,
        help="step epoch for learning rate scheduler",
    )
    parser.add_argument(
        "--include_forces",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_stresses",
        type=bool,
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--include_charges",
        type=bool,
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument("--force_loss_ratio", type=float, default=10.0)
    parser.add_argument("--stress_loss_ratio", type=float, default=0.1)
    parser.add_argument("--charge_loss_ratio", type=float, default=5.0)
    parser.add_argument("--early_stop_patience", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    main(args)