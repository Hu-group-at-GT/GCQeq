import torch
from pymatgen.core.periodic_table import Element
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius

def get_chi(max_z, device) -> torch.Tensor:
    chi = torch.zeros(max_z + 1, device=device)
    for z in range(1, max_z + 1):
        el = Element.from_Z(z)
        chi[z] = (el.ionization_energy + el.electron_affinity) / 2
    return chi

def get_J(max_z, device) -> torch.Tensor:
    J = torch.zeros(max_z + 1, device=device)
    for z in range(1, max_z + 1):
        el = Element.from_Z(z)
        J[z] = (el.ionization_energy - el.electron_affinity) / 2
    return J

def get_sigma(max_z, device) -> torch.Tensor:
    sigma = torch.zeros(max_z + 1, device=device)
    for z in range(1, max_z + 1):
        el = Element.from_Z(z).symbol
        sigma[z] = CovalentRadius.radius[el]
    return sigma