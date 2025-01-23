import torch
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from pymatgen.core import Structure
from tqdm import tqdm
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
import pdb
import numpy as np
import pandas as pd
import networkx as nx
import torch
import copy
import itertools

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env

from networkx.algorithms.components import is_connected


from torch_scatter import segment_coo, segment_csr

# from multiprocessing import Pool
from tqdm import tqdm 

from torch.utils.data import DataLoader, Dataset
import wandb
def reduce(tensor_list, keep_tensor=[]):
    reduced_tensor_list = []
    for idx, tensor in enumerate(tensor_list):
        reduced_sum = tensor.sum()/tensor.numel()
        if idx < len(keep_tensor) and keep_tensor[idx] == 0:
            reduced_tensor_list.append(reduced_sum.item())
        else:
            reduced_tensor_list.append(reduced_sum)
    return reduced_tensor_list

def generate_keep_tensor(length, keep_indices):
    keep_tensor = [0] * length
    for index in keep_indices:
        if 0 <= index < length:
            keep_tensor[index] = 1
    return keep_tensor

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=3.5, size_average=True,temperature=0.3):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            self.alpha = alpha
        self.alpha = self.alpha.type(torch.FloatTensor) 
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.temperature=temperature
     
    def forward(self, inputs, targets):
        #pdb.set_trace()
        device = inputs.device
        self.alpha = self.alpha.to(device)
        inputs=F.softmax(inputs/self.temperature,dim=-1)
        N = inputs.size(0)
        C = inputs.size(1)
        P=inputs
        #P = F.softmax(inputs, dim=1)
        class_mask = torch.zeros(N, C).to(device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids, 1.)
        alpha = self.alpha[ids.view(-1)].to(device)
        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
def argmax_with_custom_value(tensor, custom_value=-1):
    equal_mask = torch.all(tensor == tensor[:, :, 0].unsqueeze(-1), dim=-1)
    result = torch.where(equal_mask, torch.full_like(equal_mask, custom_value, dtype=torch.int64), torch.argmax(tensor, dim=-1))
    
    return result
def get_atom_indices(atom_fea):
    max_vals, atom_indices = torch.max(atom_fea.data, dim=2)
    all_same = torch.all(atom_fea.data == atom_fea.data[:, :, :1], dim=2)
    atom_indices[all_same] = -1

    return atom_indices
import random
from pymatgen.io.cif import CifParser
from io import StringIO
def count_elements_in_cifs(id_data,max=10000):
    element_counter = Counter()
    i=0

    for data in tqdm(id_data):
        i+=1
        if i>max:
            return element_counter
        else:
            cry = id_data[i]
            cif_io = StringIO(cry['cif'])
            parser = CifParser(cif_io)
            structure = parser.get_structures()[0]
            element_counter.update(structure.composition.element_composition.as_dict())
    
    return element_counter

def convert_counter_to_alpha(counter, total_elements=100,scale=1):
    alpha = np.zeros(total_elements)
    element_to_index = {element: index for index, element in enumerate(['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                                                                        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                                                                        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                                                                        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                                                                        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                                                                        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                                                                        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                                                                        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                                                                        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                                                                        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'])}
    total_count = sum(counter.values())
    for element, count in counter.items():
        index = element_to_index[element]
        alpha[index] = total_count / count
    #alpha[0]=0.001
    #alpha=[1]*len(alpha)
    print(alpha)
    alpha_tensor = torch.FloatTensor(alpha)
    return alpha_tensor*scale  


def max_matching_accuracy(target, x):
    assert target.dim() == 1 and x.dim() == 1
    assert target.size(0) == x.size(0)
    
    # Count the occurrences of each element in target and x
    unique_elements = torch.unique(torch.cat((target, x)))
    matching_count = 0
    
    for element in unique_elements:
        target_count = (target == element).sum().item()
        x_count = (x == element).sum().item()
        matching_count += min(target_count, x_count)
    
    return matching_count,target.size(0)

def tensor_to_list_based_on_mask(tensor, mask):
    b_size = tensor.shape[0] 
    result_list = []  
    
    for b in range(b_size): 
        selected_tensor = tensor[b][mask[b]] 
        result_list.append(selected_tensor)  
    
    return result_list





def int_to_element(lst):
    elements = ['H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
                'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
                'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
                'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
                'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
                'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
                'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
                'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm']
    
    return [elements[i] for i in lst]


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def crystal_position_constrain(tensor):
    transformed_tensor = tensor.clone() 
    shape=transformed_tensor.shape
    tensor_flat=transformed_tensor.reshape(-1)
    for idx in range(tensor_flat.shape[0]):
        value = tensor_flat[idx]
        if value > 0.9:
            tensor_flat[idx] = 1
        elif value < 0.1:
            tensor_flat[idx] = 0
        elif 0.45 <= value <= 0.55:
            tensor_flat[idx] = 0.5
    transformed_tensor=tensor_flat.reshape(shape)
    return transformed_tensor

def shrinkage_loss(pred,target):
    l=torch.abs(pred-target)
    shrinkage=(1+((5*(1-l))).exp()).reciprocal()
    result=(l**2)*shrinkage
    return result.sum()

def append_ones(f_mask, append_length=6):
    B, L = f_mask.shape
    ones_to_append = torch.ones(B, append_length, dtype=f_mask.dtype, device=f_mask.device)
    expanded_mask = torch.cat([f_mask, ones_to_append], dim=1)

    return expanded_mask

def convert_atom_list_to_huaxuehsi(list):
    counter=Counter(list)
    element_counts = dict(counter)
    sorted_elements = sorted(element_counts.items())
    formula = ''.join([f"{element}{count}" if count > 1 else element for element, count in sorted_elements])
    return formula

def expand_tensor(tensor, K):
    """
    Expand a boolean tensor from (B, L) to (B, K), padding with False values.
    
    Args:
    tensor (torch.Tensor): Input boolean tensor of shape (B, L).
    K (int): The target size for the second dimension.
    
    Returns:
    torch.Tensor: The expanded tensor of shape (B, K).
    """
    B, L = tensor.shape
    if L >= K:
        return tensor[:, :K]
    padding = torch.zeros((B, K - L), dtype=torch.bool,device=tensor.device)
    expanded_tensor = torch.cat((tensor, padding), dim=1)
    
    return expanded_tensor

def lattice_params_to_matrix_torch(lengths, angles):
    """Batched torch version to compute lattice matrix from params.

    lengths: torch.Tensor of shape (N, 3), unit A
    angles: torch.Tensor of shape (N, 3), unit degree
    """
    angles_r = torch.deg2rad(angles)
    coses = torch.cos(angles_r)
    sins = torch.sin(angles_r)

    val = (coses[:, 0] * coses[:, 1] - coses[:, 2]) / (sins[:, 0] * sins[:, 1])
    # Sometimes rounding errors result in values slightly > 1.
    val = torch.clamp(val, -1., 1.)
    gamma_star = torch.arccos(val)

    vector_a = torch.stack([
        lengths[:, 0] * sins[:, 1],
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 0] * coses[:, 1]], dim=1)
    vector_b = torch.stack([
        -lengths[:, 1] * sins[:, 0] * torch.cos(gamma_star),
        lengths[:, 1] * sins[:, 0] * torch.sin(gamma_star),
        lengths[:, 1] * coses[:, 0]], dim=1)
    vector_c = torch.stack([
        torch.zeros(lengths.size(0), device=lengths.device),
        torch.zeros(lengths.size(0), device=lengths.device),
        lengths[:, 2]], dim=1)

    return torch.stack([vector_a, vector_b, vector_c], dim=1)

def get_pbc_distances(
    coords,
    edge_index,
    lengths,
    angles,
    to_jimages,
    num_atoms,
    num_bonds,
    coord_is_cart=False,
    return_offsets=False,
    return_distance_vec=False,
    lattices=None
):
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)

    if coord_is_cart:
        pos = coords
    else:
        lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
        pos = torch.einsum('bi,bij->bj', coords, lattice_nodes)  # cart coords

    j_index, i_index = edge_index

    distance_vectors = pos[j_index] - pos[i_index]

    # correct for pbc
    lattice_edges = torch.repeat_interleave(lattices, num_bonds, dim=0)
    offsets = torch.einsum('bi,bij->bj', to_jimages.float(), lattice_edges)
    distance_vectors += offsets

    # compute distances
    distances = distance_vectors.norm(dim=-1)

    out = {
        "edge_index": edge_index,
        "distances": distances,
    }

    if return_distance_vec:
        out["distance_vec"] = distance_vectors

    if return_offsets:
        out["offsets"] = offsets

    return out

def radius_graph_pbc(pos, lengths, angles, natoms, radius, max_num_neighbors_threshold, device, lattices=None):
    
    # device = pos.device
    batch_size = len(natoms)
    if lattices is None:
        cell = lattice_params_to_matrix_torch(lengths, angles)
    else:
        cell = lattices
    # position of the atoms
    atom_pos = pos


    # Before computing the pairwise distances between atoms, first create a list of atom indices to compare for the entire batch
    num_atoms_per_image = natoms
    num_atoms_per_image_sqr = (num_atoms_per_image**2).long()

    # index offset between images
    index_offset = (
        torch.cumsum(num_atoms_per_image, dim=0) - num_atoms_per_image
    )

    index_offset_expand = torch.repeat_interleave(
        index_offset, num_atoms_per_image_sqr
    )
    num_atoms_per_image_expand = torch.repeat_interleave(
        num_atoms_per_image, num_atoms_per_image_sqr
    )

    # Compute a tensor containing sequences of numbers that range from 0 to num_atoms_per_image_sqr for each image
    # that is used to compute indices for the pairs of atoms. This is a very convoluted way to implement
    # the following (but 10x faster since it removes the for loop)
    # for batch_idx in range(batch_size):
    #    batch_count = torch.cat([batch_count, torch.arange(num_atoms_per_image_sqr[batch_idx], device=device)], dim=0)
    num_atom_pairs = torch.sum(num_atoms_per_image_sqr)
    index_sqr_offset = (
        torch.cumsum(num_atoms_per_image_sqr, dim=0) - num_atoms_per_image_sqr
    )
    index_sqr_offset = torch.repeat_interleave(
        index_sqr_offset, num_atoms_per_image_sqr
    )
    atom_count_sqr = (
        torch.arange(num_atom_pairs, device=device) - index_sqr_offset
    )

    # Compute the indices for the pairs of atoms (using division and mod)
    # If the systems get too large this apporach could run into numerical precision issues
    index1 = (
        torch.div(
            atom_count_sqr, num_atoms_per_image_expand, rounding_mode="floor"
        )
    ) + index_offset_expand
    index2 = (
        atom_count_sqr % num_atoms_per_image_expand
    ) + index_offset_expand
    # Get the positions for each atom
    pos1 = torch.index_select(atom_pos, 0, index1)
    pos2 = torch.index_select(atom_pos, 0, index2)

    # Calculate required number of unit cells in each direction.
    # Smallest distance between planes separated by a1 is
    # 1 / ||(a2 x a3) / V||_2, since a2 x a3 is the area of the plane.
    # Note that the unit cell volume V = a1 * (a2 x a3) and that
    # (a2 x a3) / V is also the reciprocal primitive vector
    # (crystallographer's definition).
    cross_a2a3 = torch.cross(cell[:, 1], cell[:, 2], dim=-1)
    cell_vol = torch.sum(cell[:, 0] * cross_a2a3, dim=-1, keepdim=True)
    inv_min_dist_a1 = torch.norm(cross_a2a3 / cell_vol, p=2, dim=-1)
    min_dist_a1 = (1 / inv_min_dist_a1).reshape(-1,1)

    cross_a3a1 = torch.cross(cell[:, 2], cell[:, 0], dim=-1)
    inv_min_dist_a2 = torch.norm(cross_a3a1 / cell_vol, p=2, dim=-1)
    min_dist_a2 = (1 / inv_min_dist_a2).reshape(-1,1)
    
    cross_a1a2 = torch.cross(cell[:, 0], cell[:, 1], dim=-1)
    inv_min_dist_a3 = torch.norm(cross_a1a2 / cell_vol, p=2, dim=-1)
    min_dist_a3 = (1 / inv_min_dist_a3).reshape(-1,1)
    
    # Take the max over all images for uniformity. This is essentially padding.
    # Note that this can significantly increase the number of computed distances
    # if the required repetitions are very different between images
    # (which they usually are). Changing this to sparse (scatter) operations
    # might be worth the effort if this function becomes a bottleneck.
    max_rep = torch.ones(3, dtype=torch.long, device=device)
    min_dist = torch.cat([min_dist_a1, min_dist_a2, min_dist_a3], dim=-1) # N_graphs * 3
#     reps = torch.cat([rep_a1.reshape(-1,1), rep_a2.reshape(-1,1), rep_a3.reshape(-1,1)], dim=1) # N_graphs * 3
    
    unit_cell_all = []
    num_cells_all = []

    # Tensor of unit cells
    cells_per_dim = [
        torch.arange(-rep, rep + 1, device=device, dtype=torch.float)
        for rep in max_rep
    ]
    
    unit_cell = torch.cat([_.reshape(-1,1) for _ in torch.meshgrid(cells_per_dim)], dim=-1)
    
    num_cells = len(unit_cell)
    unit_cell_per_atom = unit_cell.view(1, num_cells, 3).repeat(
        len(index2), 1, 1
    )
    unit_cell = torch.transpose(unit_cell, 0, 1)
    unit_cell_batch = unit_cell.view(1, 3, num_cells).expand(
        batch_size, -1, -1
    )

    # Compute the x, y, z positional offsets for each cell in each image
    data_cell = torch.transpose(cell, 1, 2)
    pbc_offsets = torch.bmm(data_cell, unit_cell_batch)
    pbc_offsets_per_atom = torch.repeat_interleave(
        pbc_offsets, num_atoms_per_image_sqr, dim=0
    )

    # Expand the positions and indices for the 9 cells
    pos1 = pos1.view(-1, 3, 1).expand(-1, -1, num_cells)
    pos2 = pos2.view(-1, 3, 1).expand(-1, -1, num_cells)
    index1 = index1.view(-1, 1).repeat(1, num_cells).view(-1)
    index2 = index2.view(-1, 1).repeat(1, num_cells).view(-1)
    # Add the PBC offsets for the second atom
    pos2 = pos2 + pbc_offsets_per_atom

#     # Compute the squared distance between atoms
    atom_distance_sqr = torch.sum((pos1 - pos2) ** 2, dim=1)
    atom_distance_sqr = atom_distance_sqr.view(-1)

    # Remove pairs that are too far apart
    
    
    radius_real = (min_dist.min(dim=-1)[0] + 0.01)#.clamp(max=radius)
    
    radius_real = torch.repeat_interleave(radius_real, num_atoms_per_image_sqr * num_cells)

    # print(min_dist.min(dim=-1)[0])
    
    # radius_real = radius
    
    mask_within_radius = torch.le(atom_distance_sqr, radius_real * radius_real)
    # Remove pairs with the same atoms (distance = 0.0)
    mask_not_same = torch.gt(atom_distance_sqr, 0.0001)
    mask = torch.logical_and(mask_within_radius, mask_not_same)
    index1 = torch.masked_select(index1, mask)
    index2 = torch.masked_select(index2, mask)
    unit_cell = torch.masked_select(
        unit_cell_per_atom.view(-1, 3), mask.view(-1, 1).expand(-1, 3)
    )
    unit_cell = unit_cell.view(-1, 3)
    atom_distance_sqr = torch.masked_select(atom_distance_sqr, mask)
    
    if max_num_neighbors_threshold is not None:

        mask_num_neighbors, num_neighbors_image = get_max_neighbors_mask(
            natoms=natoms,
            index=index1,
            atom_distance=atom_distance_sqr,
            max_num_neighbors_threshold=max_num_neighbors_threshold,
        )

        if not torch.all(mask_num_neighbors):
            # Mask out the atoms to ensure each atom has at most max_num_neighbors_threshold neighbors
            index1 = torch.masked_select(index1, mask_num_neighbors)
            index2 = torch.masked_select(index2, mask_num_neighbors)
            unit_cell = torch.masked_select(
                unit_cell.view(-1, 3), mask_num_neighbors.view(-1, 1).expand(-1, 3)
            )
            unit_cell = unit_cell.view(-1, 3)
            
    else:
        ones = index1.new_ones(1).expand_as(index1)
        num_neighbors = segment_coo(ones, index1, dim_size=natoms.sum())

        # Get number of (thresholded) neighbors per image
        image_indptr = torch.zeros(
            natoms.shape[0] + 1, device=device, dtype=torch.long
        )
        image_indptr[1:] = torch.cumsum(natoms, dim=0)
        num_neighbors_image = segment_csr(num_neighbors, image_indptr)

    edge_index = torch.stack((index2, index1))

    return edge_index, unit_cell, num_neighbors_image

def frac_to_cart_coords(
    frac_coords,
    lengths,
    angles,
    num_atoms,
    regularized = True,
    lattices = None
):
    if regularized:
        frac_coords = frac_coords % 1.
    if lattices is None:
        lattices = lattice_params_to_matrix_torch(lengths, angles)
    lattice_nodes = torch.repeat_interleave(lattices, num_atoms, dim=0)
    pos = torch.einsum('bi,bij->bj', frac_coords, lattice_nodes)  # cart coords

    return pos
def get_max_neighbors_mask(
    natoms, index, atom_distance, max_num_neighbors_threshold
):
    """
    Give a mask that filters out edges so that each atom has at most
    `max_num_neighbors_threshold` neighbors.
    Assumes that `index` is sorted.
    """
    device = natoms.device
    num_atoms = natoms.sum()

    # Get number of neighbors
    # segment_coo assumes sorted index
    ones = index.new_ones(1).expand_as(index)
    num_neighbors = segment_coo(ones, index, dim_size=num_atoms)
    max_num_neighbors = num_neighbors.max()
    num_neighbors_thresholded = num_neighbors.clamp(
        max=max_num_neighbors_threshold
    )

    # Get number of (thresholded) neighbors per image
    image_indptr = torch.zeros(
        natoms.shape[0] + 1, device=device, dtype=torch.long
    )
    image_indptr[1:] = torch.cumsum(natoms, dim=0)
    num_neighbors_image = segment_csr(num_neighbors_thresholded, image_indptr)

    # If max_num_neighbors is below the threshold, return early
    if (
        max_num_neighbors <= max_num_neighbors_threshold
        or max_num_neighbors_threshold <= 0
    ):
        mask_num_neighbors = torch.tensor(
            [True], dtype=bool, device=device
        ).expand_as(index)
        return mask_num_neighbors, num_neighbors_image

    # Create a tensor of size [num_atoms, max_num_neighbors] to sort the distances of the neighbors.
    # Fill with infinity so we can easily remove unused distances later.
    distance_sort = torch.full(
        [num_atoms * max_num_neighbors], np.inf, device=device
    )

    # Create an index map to map distances from atom_distance to distance_sort
    # index_sort_map assumes index to be sorted
    index_neighbor_offset = torch.cumsum(num_neighbors, dim=0) - num_neighbors
    index_neighbor_offset_expand = torch.repeat_interleave(
        index_neighbor_offset, num_neighbors
    )
    index_sort_map = (
        index * max_num_neighbors
        + torch.arange(len(index), device=device)
        - index_neighbor_offset_expand
    )
    distance_sort.index_copy_(0, index_sort_map, atom_distance)
    distance_sort = distance_sort.view(num_atoms, max_num_neighbors)

    # Sort neighboring atoms based on distance
    distance_sort, index_sort = torch.sort(distance_sort, dim=1)
    # Select the max_num_neighbors_threshold neighbors that are closest
    distance_real_cutoff = distance_sort[:,max_num_neighbors_threshold].reshape(-1,1).expand(-1,max_num_neighbors) + 0.01
    
    mask_distance = distance_sort < distance_real_cutoff
    
    index_sort = index_sort + index_neighbor_offset.view(-1, 1).expand(
        -1, max_num_neighbors
    )
    
    
    # Remove "unused pairs" with infinite distances
    mask_finite = torch.isfinite(distance_sort)
#     index_sort = torch.masked_select(index_sort, mask_finite)
    index_sort = torch.masked_select(index_sort, mask_finite & mask_distance)
    
    num_neighbor_per_node = (mask_finite & mask_distance).sum(dim=-1)
    num_neighbors_image = segment_csr(num_neighbor_per_node, image_indptr)
    

    # At this point index_sort contains the index into index of the
    # closest max_num_neighbors_threshold neighbors per atom
    # Create a mask to remove all pairs not in index_sort
    mask_num_neighbors = torch.zeros(len(index), device=device, dtype=bool)
    mask_num_neighbors.index_fill_(0, index_sort, True)

    return mask_num_neighbors, num_neighbors_image
def repeat_blocks(
    sizes,
    repeats,
    continuous_indexing=True,
    start_idx=0,
    block_inc=0,
    repeat_inc=0,
):
    
    assert sizes.dim() == 1
    assert all(sizes >= 0)

    # Remove 0 sizes
    sizes_nonzero = sizes > 0
    if not torch.all(sizes_nonzero):
        assert block_inc == 0  # Implementing this is not worth the effort
        sizes = torch.masked_select(sizes, sizes_nonzero)
        if isinstance(repeats, torch.Tensor):
            repeats = torch.masked_select(repeats, sizes_nonzero)
        if isinstance(repeat_inc, torch.Tensor):
            repeat_inc = torch.masked_select(repeat_inc, sizes_nonzero)

    if isinstance(repeats, torch.Tensor):
        assert all(repeats >= 0)
        insert_dummy = repeats[0] == 0
        if insert_dummy:
            one = sizes.new_ones(1)
            zero = sizes.new_zeros(1)
            sizes = torch.cat((one, sizes))
            repeats = torch.cat((one, repeats))
            if isinstance(block_inc, torch.Tensor):
                block_inc = torch.cat((zero, block_inc))
            if isinstance(repeat_inc, torch.Tensor):
                repeat_inc = torch.cat((zero, repeat_inc))
    else:
        assert repeats >= 0
        insert_dummy = False

    # Get repeats for each group using group lengths/sizes
    r1 = torch.repeat_interleave(
        torch.arange(len(sizes), device=sizes.device), repeats
    )

    # Get total size of output array, as needed to initialize output indexing array
    N = (sizes * repeats).sum()

    # Initialize indexing array with ones as we need to setup incremental indexing
    # within each group when cumulatively summed at the final stage.
    # Two steps here:
    # 1. Within each group, we have multiple sequences, so setup the offsetting
    # at each sequence lengths by the seq. lengths preceding those.
    id_ar = torch.ones(N, dtype=torch.long, device=sizes.device)
    id_ar[0] = 0
    insert_index = sizes[r1[:-1]].cumsum(0)
    insert_val = (1 - sizes)[r1[:-1]]

    if isinstance(repeats, torch.Tensor) and torch.any(repeats == 0):
        diffs = r1[1:] - r1[:-1]
        indptr = torch.cat((sizes.new_zeros(1), diffs.cumsum(0)))
        if continuous_indexing:
            # If a group was skipped (repeats=0) we need to add its size
            insert_val += segment_csr(sizes[: r1[-1]], indptr, reduce="sum")

        # Add block increments
        if isinstance(block_inc, torch.Tensor):
            insert_val += segment_csr(
                block_inc[: r1[-1]], indptr, reduce="sum"
            )
        else:
            insert_val += block_inc * (indptr[1:] - indptr[:-1])
            if insert_dummy:
                insert_val[0] -= block_inc
    else:
        idx = r1[1:] != r1[:-1]
        if continuous_indexing:
            # 2. For each group, make sure the indexing starts from the next group's
            # first element. So, simply assign 1s there.
            insert_val[idx] = 1

        # Add block increments
        insert_val[idx] += block_inc

    # Add repeat_inc within each group
    if isinstance(repeat_inc, torch.Tensor):
        insert_val += repeat_inc[r1[:-1]]
        if isinstance(repeats, torch.Tensor):
            repeat_inc_inner = repeat_inc[repeats > 0][:-1]
        else:
            repeat_inc_inner = repeat_inc[:-1]
    else:
        insert_val += repeat_inc
        repeat_inc_inner = repeat_inc

    # Subtract the increments between groups
    if isinstance(repeats, torch.Tensor):
        repeats_inner = repeats[repeats > 0][:-1]
    else:
        repeats_inner = repeats
    insert_val[r1[1:] != r1[:-1]] -= repeat_inc_inner * repeats_inner

    # Assign index-offsetting values
    id_ar[insert_index] = insert_val

    if insert_dummy:
        id_ar = id_ar[1:]
        if continuous_indexing:
            id_ar[0] -= 1

    # Set start index now, in case of insertion due to leading repeats=0
    id_ar[0] += start_idx

    # Finally index into input array for the group repeated o/p
    res = id_ar.cumsum(0)
    return res
def batch_crystals(features, atoms_per_crystal):
    max_atoms = max(atoms_per_crystal)
    batched_features = []
    start_idx = 0
    
    for count in atoms_per_crystal:
        end_idx = start_idx + count
        crystal_features = features[start_idx:end_idx].clone()  # Use clone to ensure no issues with in-place ops elsewhere affecting this tensor
        if count < max_atoms:
            padding = torch.zeros((max_atoms - count, features.shape[1]), dtype=features.dtype, device=features.device)
            crystal_features = torch.cat((crystal_features, padding), dim=0)
        batched_features.append(crystal_features)
        start_idx = end_idx
    
    batched_features = torch.stack(batched_features)
    
    return batched_features

def batch_lattice_to_matrix_torch(params):
    
    B = params.size(0)
    matrices = []

    for i in range(B):
        a, b, c, alpha, beta, gamma = params[i].tolist()
        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
        matrix = torch.tensor(lattice.matrix, dtype=torch.float32)
        matrices.append(matrix)
    matrices_tensor = torch.stack(matrices, dim=0).to(params.device)
    return matrices_tensor

def matrix_to_lattice_params(batch_matrices):

    B = batch_matrices.size(0)
    params = []

    for i in range(B):
        matrix = batch_matrices[i].detach().cpu().numpy()
        lattice = Lattice(matrix)
        a, b, c = lattice.lengths
        alpha, beta, gamma = lattice.angles
        params.append([a, b, c, alpha, beta, gamma])
    params_tensor = torch.tensor(params, dtype=torch.float32).to(batch_matrices.device)
    return params_tensor


import os
import shutil

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path) 
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')