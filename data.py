from __future__ import print_function, division

import csv
import functools
import json
import os
import random
import warnings
from utils import *
import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler
from pymatgen.analysis.local_env import CrystalNN
import pandas as pd
from pymatgen.io.cif import CifParser
from io import StringIO
import ast
def get_train_val_test_loader(
    dataset,
    collate_fn=default_collate,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=False,
    num_workers=1,
    pin_memory=False,
    **kwargs,
):
    
    total_size = len(dataset)
    indices = list(range(total_size))
    train_size = int(train_ratio * total_size)
    valid_size = int(val_ratio * total_size)
    test_size = total_size - train_size - valid_size
    train_sampler = SubsetRandomSampler(indices[:train_size])
    val_sampler = SubsetRandomSampler(indices[-(valid_size + test_size) : -test_size])
    if return_test:
        test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader

def collate_pool(dataset_list):

    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx = []
    batch_cif_ids = []
    masks_atom = []
    masks_ad = []
    masks_coords = []
    padded_batch_atom_fea = []
    padded_ad_matrix = []
    padded_coords_tensor = []
    batch_lattice = []
    batch_proper=[]
    batch_force=[]
    crystals = []
    space_groups=[]
    base_idx = 0
    max_N = max(item[0]['atom_fea'].shape[0] for item in dataset_list)
    for i, (data_dict, cif_id) in enumerate(dataset_list):

        atom_fea = data_dict['atom_fea']
        nbr_fea = data_dict['nbr_fea']
        nbr_fea_idx = data_dict['nbr_fea_idx']
        ad_matrix = data_dict['ad_matrix']
        frac_coords_tensor = data_dict['frac_coords_tensor']
        lattice = data_dict['lattice']
        crystal = data_dict['crystal']
        proper=data_dict['property']
        space_group=data_dict['space_group']
        n_i, l = atom_fea.shape  # number of atoms for this crystal
        padded_atom_fea = torch.zeros(max_N, l)
        padded_atom_fea[:n_i, :] = atom_fea
        mask = torch.zeros(max_N, dtype=torch.bool)
        mask[:atom_fea.shape[0]] = True
        masks_atom.append(mask)
        padded_batch_atom_fea.append(padded_atom_fea)

        ad_matrix_padded = torch.zeros(max_N, max_N)
        ad_matrix_padded[:n_i, :n_i] = ad_matrix
        ad_mask = torch.zeros(max_N, max_N, dtype=torch.bool)
        ad_mask[:n_i, :n_i] = True
        masks_ad.append(ad_mask)
        padded_ad_matrix.append(ad_matrix_padded)

        frac_coords_padded = torch.zeros(max_N, 3)
        frac_coords_padded[:n_i, :] = frac_coords_tensor
        coords_mask = torch.zeros(max_N, 3, dtype=torch.bool)
        coords_mask[:n_i, :] = True
        masks_coords.append(coords_mask)
        padded_coords_tensor.append(frac_coords_padded)
        batch_proper.append(proper)
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_lattice.append(lattice)
        batch_cif_ids.append(cif_id)
        crystals.append(crystal)
        space_groups.append(space_group)
        base_idx += n_i

    result_dict = {
        'atom_fea': NestedTensor(
            data=torch.stack(padded_batch_atom_fea, dim=0),
            mask=torch.stack(masks_atom, dim=0),
        ),
        'ad_matrix': NestedTensor(
            data=torch.stack(padded_ad_matrix, dim=0),
            mask=torch.stack(masks_ad, dim=0),
        ),
        'frac_coords_tensor': NestedTensor(
            data=torch.stack(padded_coords_tensor, dim=0),
            mask=torch.stack(masks_coords, dim=0),
        ),
        'lattice': torch.stack(batch_lattice, dim=0),
        'crystals': crystals,
        'nbr_fea': torch.cat(batch_nbr_fea, dim=0),
        'nbr_fea_idx': torch.cat(batch_nbr_fea_idx, dim=0),
        'crystal_atom_idx': crystal_atom_idx,
        'property':torch.tensor(batch_proper),
        'space_group':torch.tensor(space_groups)
    }

    return result_dict, batch_cif_ids


class NestedTensor:
    def __init__(self, data=None, mask=None):
        self.data = data if data is not None else []
        self.mask = mask if mask is not None else []

    def add_tensor(self, tensor, mask):
        self.data.append(tensor)
        self.mask.append(mask)

    def __getitem__(self, index):
        return self.data[index], self.mask[index]

    def shape(self):
        return self.data.shape

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return zip(self.data, self.mask)

    def __repr__(self):
        return f"NestedTensor(data={self.data}, mask={self.mask})"


class GaussianDistance(object):
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer(object):
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):


    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super(AtomCustomJSONInitializer, self).__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


class CIFData(Dataset):
    

    def __init__(
        self,
        root_dir,
        max_num_nbr=12,
        radius=12,
        dmin=0,
        step=0.2,
        random_seed=123,
        nb_cut=0.5,
        max_N=64,
        keep=False,
        num_samples=10000,
        proper='formation_energy',
        force=False,
        select_num=None,
        base_id=None,
    ):
        self.root_dir = root_dir
        self.proper=proper
        if self.proper=='formation_energy':
            self.proper='formation_energy_per_atom'
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        if force:
            self.id_prop_data = pd.read_csv(os.path.join(root_dir,'data_with_forces.csv'))
        else:
            self.id_prop_data = pd.read_csv(os.path.join(root_dir,'data.csv'))
        self.id_prop_data=self.id_prop_data.to_dict(orient='records')

        if select_num is not None:
            random.seed(random_seed)
            self.id_prop_data = random.sample(self.id_prop_data, select_num)
        random.seed(random_seed)
        random.shuffle(self.id_prop_data)

        # for inference
        if base_id is not None:
            if base_id + 100 < len(self.id_prop_data):
                self.id_prop_data = self.id_prop_data[base_id: base_id+100]
                self.mark = 0
            else:
                self.id_prop_data = self.id_prop_data[base_id-100: base_id+1]
                self.mark = -1
        else:
            self.mark = random.randint(0, len(self.id_prop_data)-1)

        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        # total_size = len(self.id_prop_data)
        # valid_size = int(0.2* total_size)
        # test_size = int(0.2* total_size)
        # train_size = total_size - valid_size - test_size
        # train_data = self.id_prop_data[:train_size]
        # val_test_data = self.id_prop_data[-(valid_size + test_size):]
        # df = pd.DataFrame(train_data)
        # csv_file_path = 'mp_train_data.csv'
        # df.to_csv(csv_file_path, index=False)
        
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)
        self.nb_cut = nb_cut
        self.max_N = max_N
        self.removed_crystals_count = 0       
        if keep:
            from tqdm import tqdm

            self.id_prop_data = [
                cry for cry in tqdm(self.id_prop_data) if self._keep_crystal(cry)
            ]
            print(
                f"Removed {self.removed_crystals_count} crystals with more than {max_N} atoms."
            )
    
        self.norm = self.calculate_average_lattice_parameters(num_samples)
        self.norm_properties=self.calculate_average_properties(num_samples,self.proper)
        # pdb.set_trace()
        self.element_counter = count_elements_in_cifs(self.id_prop_data,max=10)
        self.angular_encoder = AngularEncoder(64)

    def calculate_average_properties(self, num_samples,proper_name):
        proper_list = []
        from tqdm import tqdm

        for i, cry in tqdm(enumerate(self.id_prop_data)):
            if i >= num_samples:
                break
            proper=cry[proper_name]
            proper_list.append(proper)
        proper_list = torch.tensor(proper_list)
        return Normalizer(proper_list)

    def calculate_average_lattice_parameters(self, num_samples):
        lattice_parameters = []
        from tqdm import tqdm

        for i, cry in tqdm(enumerate(self.id_prop_data)):
            if i >= num_samples:
                break
            cif_io = StringIO(cry['cif'])
            parser = CifParser(cif_io)
            structure = parser.get_structures()[0]
            lattice = structure.lattice
            lattice_parameters.append(
                [
                    lattice.a,
                    lattice.b,
                    lattice.c,
                    lattice.alpha,
                    lattice.beta,
                    lattice.gamma,
                ]
            )
        lattice_parameters = torch.tensor(lattice_parameters)
        return Normalizer(lattice_parameters)

    def __len__(self):
        return len(self.id_prop_data)

    def _keep_crystal(self, cry):
        cif=cry['cif']
        cif_io = StringIO(cif)
        parser = CifParser(cif_io)
        crystal = parser.get_structures()[0]
        if len(crystal) > self.max_N:
            self.removed_crystals_count += 1
            return False
        return True

    
    
    @functools.lru_cache(maxsize=None)  # Cache loaded structures
    def __getitem__(self, idx):
        cry = self.id_prop_data[idx]
        cif_io = StringIO(cry['cif'])
        parser = CifParser(cif_io)
        crystal = parser.get_structures()[0]
        try:
            space_group_info = crystal.get_space_group_info()
            spacegroup_number = space_group_info[1]
        except:
            spacegroup_number = 1  # Default to P1 if space group information is not available
        cif_id=cry['material_id']
        nn = CrystalNN()
        n_atoms = len(crystal)  
        distances = [[0.0] * n_atoms for _ in range(n_atoms)]
        lattice = crystal.lattice
        a_length, b_length, c_length = lattice.abc
        alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
        lattice = [a_length, b_length, c_length, alpha, beta, gamma]
        cutl = self.nb_cut * (a_length + b_length + c_length) / 3
        for i, site in enumerate(crystal):
            for j, other_site in enumerate(crystal):
                distances[i][j] = crystal.get_distance(i, j)
        distances_tensor = torch.tensor(distances, dtype=torch.float32)
        ad_matrix = distances_tensor < cutl

        frac_coords = distances = [[0.0] * 3 for _ in range(n_atoms)]
        for i, site in enumerate(crystal):
            frac_coords[i] = site.frac_coords
        frac_coords_array = np.array(frac_coords, dtype=np.float32)
        frac_coords_tensor = (torch.tensor(frac_coords_array) - 0.5) * 2
        frac_coords_tensor = torch.clamp(frac_coords_tensor, min=-1, max=1)
        atom_fea = np.vstack(
            [
                self.ari.get_atom_fea(crystal[i].specie.number)
                for i in range(len(crystal))
            ]
        )
        atom_fea = torch.Tensor(atom_fea)
        all_nbrs = crystal.get_all_neighbors(self.radius, include_index=True)
        all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
        nbr_fea_idx, nbr_fea, angle_fea_x ,angle_fea_y,angle_fea_z= [], [], [],[],[]
        for atom_idx, nbr in enumerate(all_nbrs):
            atom = crystal[atom_idx]
            if len(nbr) < self.max_num_nbr:
                warnings.warn(' not find enough neighbors to build graph. '
                              'If it happens frequently, consider increase '
                              'radius.')
                nbr_fea_idx.append(list(map(lambda x: x[2], nbr)) +
                                   [0] * (self.max_num_nbr - len(nbr)))
                nbr_fea.append(list(map(lambda x: x[1], nbr)) +
                               [self.radius + 1.] * (self.max_num_nbr -
                                                     len(nbr)))
                angles_x, angles_y, angles_z = [], [], []
                for site in nbr:
                    vec = site.coords - atom.coords
                    angle_x = angle_between_sites(np.array([1, 0, 0]), vec)
                    angle_y = angle_between_sites(np.array([0, 1, 0]), vec)
                    angle_z = angle_between_sites(np.array([0, 0, 1]), vec)
                    angles_x.append(angle_x)
                    angles_y.append(angle_y)
                    angles_z.append(angle_z)
                for i in range(self.max_num_nbr - len(nbr)):
                    angles_x.append(0)
                    angles_y.append(0)
                    angles_z.append(0)
                angle_fea_x.append(self.angular_encoder.encode(angles_x))
                angle_fea_y.append(self.angular_encoder.encode(angles_y))
                angle_fea_z.append(self.angular_encoder.encode(angles_z))
            else:
                nbr_fea_idx.append(list(map(lambda x: x[2],
                                            nbr[:self.max_num_nbr])))
                nbr_fea.append(list(map(lambda x: x[1],
                                        nbr[:self.max_num_nbr])))
                angles_x,angles_y,angles_z = [],[],[]
                for site in nbr[:self.max_num_nbr]:
                    vec = site.coords - atom.coords
                    angle_x = angle_between_sites(np.array([1, 0, 0]), vec)
                    angle_y = angle_between_sites(np.array([0, 1, 0]), vec)
                    angle_z = angle_between_sites(np.array([0, 0, 1]), vec)
                    angles_x.append(angle_x)
                    angles_y.append(angle_y)
                    angles_z.append(angle_z)
                angle_fea_x.append(self.angular_encoder.encode(angles_x))
                angle_fea_y.append(self.angular_encoder.encode(angles_y))
                angle_fea_z.append(self.angular_encoder.encode(angles_z))
        nbr_fea_idx, nbr_fea, angle_fea_x,angle_fea_y,angle_fea_z = np.array(nbr_fea_idx), np.array(nbr_fea), np.array(angle_fea_x),np.array(angle_fea_y),np.array(angle_fea_z)
        nbr_fea = self.gdf.expand(nbr_fea)
        angle_fea_x = torch.Tensor(angle_fea_x)
        angle_fea_y = torch.Tensor(angle_fea_y)
        angle_fea_z = torch.Tensor(angle_fea_z)     
        atom_fea = torch.Tensor(atom_fea)
        nbr_fea = torch.Tensor(nbr_fea)
        nbr_fea_idx = torch.LongTensor(nbr_fea_idx) 
        # Concatenate distance features and angle features
        nbr_fea = torch.cat((nbr_fea, angle_fea_x,angle_fea_y,angle_fea_z), dim=-1)
        
        lattice = torch.Tensor(lattice)
        if 'formation_energy' in self.proper:
            proper=torch.tensor(cry['formation_energy_per_atom'])
        else:
            proper= torch.tensor(cry[self.proper])
        data_dict = {
            'atom_fea': atom_fea,
            'nbr_fea': nbr_fea,
            'nbr_fea_idx': nbr_fea_idx,
            'ad_matrix': ad_matrix,
            'frac_coords_tensor': frac_coords_tensor,
            'lattice': lattice,
            'crystal': crystal,
            'property': proper,
            'space_group':spacegroup_number,
        }
        
        return data_dict, cif_id


class Normalizer(object):
    """Normalize a Tensor to [-1, 1] range and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the min and max."""
        self.min = torch.min(tensor,dim=0)[0]
        self.max = torch.max(tensor,dim=0)[0]

    def norm(self, tensor):
        #pdb.set_trace()
        """Normalize the tensor to [-1, 1] range."""
        # Ensure min and max are on the same device as the input tensor
        self.min = self.min.to(tensor.device)
        self.max = self.max.to(tensor.device)
        # Normalize tensor to [0, 1] then scale to [-1, 1]
        return ((tensor - self.min) / (self.max - self.min + 1e-10)) * 2 - 1

    def denorm(self, normed_tensor):
        """Restore the tensor from [-1, 1] range to its original range."""
        self.min = self.min.to(normed_tensor.device)
        self.max = self.max.to(normed_tensor.device)
        # First shift from [-1, 1] to [0, 1], then to original
        return ((normed_tensor + 1) / 2) * (self.max - self.min) + self.min

class AngularEncoder:
    def __init__(self, bins):
        self.bins = bins

    def encode(self, angles):
        angles = np.array(angles)
        encoded_angles = np.zeros((len(angles), self.bins), dtype=np.float32)
        bin_width = 2 * np.pi / self.bins

        for i, angle in enumerate(angles):
            encoded_bin = int(angle // bin_width)
            encoded_angles[i, encoded_bin] = 1

        return encoded_angles

def angle_between_sites(v1, v2):
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm_product)
    return angle