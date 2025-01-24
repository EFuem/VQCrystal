import argparse
import os
import shutil

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pickle import NONE
import warnings
from random import sample
import numpy as np
from pymatgen.io.cif import CifWriter
import torch
import random
import pdb
import yaml
from tqdm import tqdm
from data import CIFData, collate_pool, get_train_val_test_loader
from model.model2 import GraphVAE
from utils import *
from pymatgen.core.structure import Structure
from geneticalgorithm import geneticalgorithm as ga
from datetime import datetime
from pymatgen.io.cif import CifWriter
import time
import pandas as pd
import torch
from collections import deque
from tqdm import tqdm

###openlam###
from lam_optimize.relaxer import Relaxer
from pathlib import Path
from lam_optimize.main import relax_run

import warnings
import warnings

warnings.filterwarnings("ignore")

current_folder = os.path.dirname(os.path.abspath(__file__))
temp_folder = current_folder
main_folder = current_folder


parser = argparse.ArgumentParser(description="basic argument for crystalvae")
parser.add_argument("--dataset", default="mp_20", type=str, required=False)
# parser.add_argument("--data_path", default=os.path.join(current_folder, "data/mp_20"), type=str, required=False)
# parser.add_argument("--model_path", default=os.path.join(current_folder, "weight/mp_20_model_weights_128_4_.pth"), type=str, required=False)
# parser.add_argument("--config_path", default=os.path.join(current_folder, "config/config_mp_20.yaml"), type=str, required=False)
parser.add_argument("--load", default=True, type=bool)
parser.add_argument("--base_id", default=-1, type=int, required=False)
parser.add_argument("--machine_id", default="machine_0", type=str, required=False)

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

match args.dataset:
    case "c2db":
        args.data_path = os.path.join(main_folder, "data/c2db")
        args.model_path = os.path.join(main_folder, "ckpt/c2db/c2db_model_weights_64_4_.pth")
        args.config_path = os.path.join(main_folder, "config/config_c2db.yaml")
    case "carbon_24":
        args.data_path = os.path.join(main_folder, "data/carbon_24")
        args.model_path = os.path.join(main_folder, "ckpt/carbon_24/carbon_24_model_weights_32_2_.pth")
        args.config_path = os.path.join(main_folder, "config/config_carbon_24.yaml")
    case "mp_20":
        args.data_path = os.path.join(main_folder, "data/mp_20")
        args.model_path = os.path.join(main_folder, "ckpt/mp_20/mp_20_model_weights_128_4_.pth")
        args.config_path = os.path.join(main_folder, "config/config_mp_20.yaml")
    case "carbon_24":
        args.data_path = os.path.join(main_folder, "data/perov_5")
        args.model_path = os.path.join(main_folder, "ckpt/perov_5/perov_5_model_weights_32_2_.pth")
        args.config_path = os.path.join(main_folder, "config/config_perov_5.yaml")
    case _:
        print("no match")

# energy_model = load_model("Eform_MP_2019")
 # using default mace model
# relaxer = Relaxer("mace")
relaxer = Relaxer(Path(os.path.join(main_folder, "dp0529.pth")), optimizer="BFGS")

def predict_energy(structure):
    structure.requires_grad = True
    calculator = relaxer.calculator
    adaptor = relaxer.ase_adaptor
    structure = adaptor.get_atoms(structure)
    structure.set_calculator(calculator)
    energy = structure.get_potential_energy()
    # force = structure.get_forces()
    return energy

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)

def load_model_and_data(base_id):
    '''
    return model and dataset, model is on cuda while data is on cpu
    unless during matrix computation, all data should be on cpu
    data is in list
    '''
    
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["cuda"] = torch.cuda.is_available()
    config["data_path"] = args.data_path
    config["batch_size"]=1
    device = torch.device("cuda" if config["cuda"] else "cpu")
    
    # if select:
    #     # 在这里就已经把数据切好了，只取一个
    #     df = pd.read_csv('./data/mp_20/data.csv')
    #     if base is None:
    #         base = random.randint(0, len(df)-1)
    #     small_df = df.iloc[base]
    #     small_df.to_csv('./data/mp_small/data.csv', index=False)
    if args.data_path == os.path.join(temp_folder, "data/mp_20"):
        # dataset = CIFData(config["data_path"], keep=True, max_N=config["k"], 
        #                 base_id=base_id, proper=config.get("property",'formation energy'))
        dataset = CIFData(config["data_path"], keep=True, max_N=config["k"], 
                        base_id=base_id)
    else:
        dataset = CIFData(config["data_path"], keep=True, max_N=config["k"], 
                        base_id=base_id)
    # dataset = dataset[:1000]
    element_counter = dataset.element_counter
    normalizer = dataset.norm
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=config["batch_size"],
        train_ratio=1,
        num_workers=config["workers"],
        val_ratio=0,
        test_ratio=0,
        pin_memory=False,
        train_size=NONE,
        val_size=NONE,
        test_size=NONE,
        return_test=True,
    )
    loss = FocalLoss(
        class_num=config["class_dim"]
    )
    model_config = {
        "loss": loss,
        "class_dim": config["class_dim"],
        "pos_dim": config["pos_dim"],
        "lattice_dim": config["lattice_dim"],
        "hidden_dim": config["hidden_dim"],
        "latent_dim": config["latent_dim"],
        "k": config["k"],
        "encoder1_dim": config["encoder1_dim"],
        "num_encoder2_layers": config["num_encoder2_layers"],
        "decoder2_dims": config["decoder2_dims"],
        "decoder1_dims": config["decoder1_dims"],
        "decoder_lattice_dims": config["decoder_lattice_dims"],
        "decoder_lattice_dims2": config["decoder_lattice_dims2"],
        "codebooksize1": config["codebooksize1"],
        "codebooksize2": config["codebooksize2"],
        "codebooksizel": config["codebooksizel"],
        "num_quantizers1": config["num_quantizers1"],
        "num_quantizers2": config["num_quantizers2"],
        "num_quantizersl": config["num_quantizersl"],
        "sample_codebook_temp": config["sample_codebook_temp"],
        "normalizer": normalizer,
    }


    model = GraphVAE(**model_config)
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    if args.load:
        print("loading model weights........")
        model.load_state_dict(
            torch.load(args.model_path, map_location=device)['model_state_dict'], strict=False
        )
        print("Load OK!")
    
    model.eval()
   
    batches=[]
    with tqdm(train_loader) as pbar:
        for batch in pbar:
            '''
            batch_dict, _ = batch
            atom_fea = batch_dict['atom_fea']
            A = batch_dict['ad_matrix']
            coords = batch_dict['frac_coords_tensor']
            lattice = batch_dict['lattice']
            crystals = batch_dict['crystals']
            nbr_fea = batch_dict['nbr_fea']
            nbr_fea_idx = batch_dict['nbr_fea_idx']
            crystal_atom_idx = batch_dict['crystal_atom_idx']
            properties=batch_dict['property'].to(device)
            F = torch.cat((atom_fea.data, coords.data), dim=2)
            flat_mask = atom_fea.mask  # (B, N) (T,T,T,F,F)
            F = F.to(device)
            A = A.data.to(device)
            nbr_fea, nbr_fea_idx = nbr_fea.to(device), nbr_fea_idx.to(device)
            crystal_atom_idx = [
                crys_idx.to(device) for crys_idx in crystal_atom_idx
            ]
            flat_mask = flat_mask.to(device)
            lattice = lattice.to(device)
            '''
            batches.append(batch)
    print(len(batches))

    return model, batches[dataset.mark]

def parse_batch(model, batch):
    '''
    batch size is 1 in default

    local_indices: (1, N, 8), remains on cuda all the time to speed up
    global_indices: (8,) numpy on cpu
    '''
    batch_dict, _ = batch
    atom_fea = batch_dict['atom_fea']
    A = batch_dict['ad_matrix']
    coords = batch_dict['frac_coords_tensor']
    lattice = batch_dict['lattice']
    crystals = batch_dict['crystals']
    nbr_fea = batch_dict['nbr_fea']
    nbr_fea_idx = batch_dict['nbr_fea_idx']
    crystal_atom_idx = batch_dict['crystal_atom_idx']
    properties=batch_dict['property'].to(args.device)
    F = torch.cat((atom_fea.data, coords.data), dim=2)
    flat_mask = atom_fea.mask  # (B, N) (T,T,T,F,F)
    F = F.to(args.device)
    A = A.data.to(args.device)
    nbr_fea, nbr_fea_idx = nbr_fea.to(args.device), nbr_fea_idx.to(args.device)
    crystal_atom_idx = [
        crys_idx.to(args.device) for crys_idx in crystal_atom_idx
    ]
    flat_mask = flat_mask.to(args.device)
    lattice = lattice.to(args.device)

    indices = model.generate_encode(
                    F, A, flat_mask, lattice, nbr_fea, nbr_fea_idx, crystal_atom_idx
                )
    local_indices, global_indices = indices
    global_indices = global_indices.squeeze(0).squeeze(0)
    global_indices = global_indices.cpu().numpy()
    return local_indices, global_indices, flat_mask

def recon_crystal(local_indices, global_indices, f_mask, model, local=True, return_vector = False):
    local_feature = model.vq1.get_output_from_indices(local_indices) # torch.Size([1, 10, 64])
    global_indices = torch.from_numpy(global_indices).unsqueeze(0).unsqueeze(0).to(args.device)
    global_feature = model.vq2.get_output_from_indices(global_indices).squeeze(0)
    z_q = (local_feature, global_feature)
    crystals = model.generate_decode(z_q, f_mask, local=local)
    if return_vector:
        return crystals[0], global_feature.reshape(-1).cpu().numpy()
    return crystals[0]

@torch.no_grad()
def recon_crystal_batch(local_indices, global_indices_2d, f_mask, model, local):
    # local indices: (1, N,  8)
    B = global_indices_2d.shape[0]
    local_feature = model.vq1.get_output_from_indices(local_indices).repeat(B, 1, 1)
    global_indices_2d = torch.from_numpy(global_indices_2d).unsqueeze(1).to(args.device)
    global_feature = model.vq2.get_output_from_indices(global_indices_2d).squeeze(0)
    global_feature = global_feature.squeeze(1)
    z_q = (local_feature, global_feature) # torch.Size([100, 4, 64])
    f_mask = f_mask.repeat(B, 1)
    crystals = model.generate_decode(z_q, f_mask, local=local)
    return crystals

######################################################################
def optimize_crystals(cif_folder, save_folder, save_threshold, check_convergence=True):
    cif_folder_path = Path(cif_folder)
    filter_same = False if args.data_path == os.path.join(current_folder, "data/carbon_24") else True
    res_df = relax_run(
        cif_folder_path,
        relaxer,
        fmax=0.04,
        steps=100,
        check_convergence=check_convergence,
        check_duplicate=False,
        machine_id=args.machine_id,
        filter_same=filter_same,
        save_folder=save_folder
    )
    new_collections = []
    for i in range(len(res_df)):
        e = res_df['final_energy'][i]
        if e >= save_threshold[0] and e <= save_threshold[1]:
            new_collections.append((e, Structure.from_dict(res_df['final_structure'][i])))
    # for e, structure in new_collections:
    #     cif_writer = CifWriter(structure)
    #     chemical_formula = structure.composition.formula
    #     cif_writer.write_file(os.path.join(save_folder, f"{e:.4f}_{chemical_formula}.cif"))
    return len(new_collections)
    
def GA_sample(base = None, keep = 100, base_size = 45238, dimension=8, varbound=None, GA_params=None, 
            init=True, save = True, save_threshold=0.045, seed = None, batch=False, local = True, post_optimize=False, check_convergence=True):
    # if seed is not None:
    #     set_seed(seed)
    # else:
    #     set_seed(int(time.time() % 1000))
    # if base < 0:
    #     base = random.randint(0, base_size-1)
    model, data = load_model_and_data(base)  # accelerate this process
    
    # data = batches[base]
    if save:
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d-%H-%M-%S")
        base_folder = os.path.join(temp_folder, f"denovo/{args.machine_id}_{args.base_id}_{timestamp_str}/base")
        os.makedirs(base_folder, exist_ok=True)
        batch_dict, _ = data
        base_crystal = batch_dict['crystals'][0]
        cif_writer = CifWriter(base_crystal)
        chemical_formula = base_crystal.composition.formula
        cif_writer.write_file(os.path.join(base_folder, f"{chemical_formula}.cif"))
    local_indices, _, f_mask = parse_batch(model, data)
    collection = {}
    if init:
        population_size = GA_params['population_size']
        indexes = [random.randint(0, base_size-1) for _ in range(population_size)]
        init_population = []
        for index in indexes:

            data = batches[index]
            _, global_indices, _ = parse_batch(model, data)
            init_population.append(global_indices)
        init_population = np.array(init_population)
    
    def fitness_function(global_indices):
        # 把global_indices的类型转换为numpy long
        # global indices np (8, )
        global_indices = global_indices.astype(np.int64)
        
        structure = recon_crystal(local_indices, global_indices, f_mask, model, local)
        try:
            energy = predict_energy(structure)
        except:
            energy = 10
        if energy >= save_threshold[0] and energy <= save_threshold[1]:
            chemical_formula = structure.composition.formula
            if chemical_formula not in collection.keys():
                collection[chemical_formula] = (structure, energy)
            else:
                if collection[chemical_formula][1] > energy:
                    collection[chemical_formula] = (structure, energy)
        return energy
    
    def fitness_function_batch(global_indices_2d):
        # 把global_indices的类型转换为numpy long
        # global indices (100,  8)
        global_indices_2d = global_indices_2d.astype(np.int64)
        
        structures = recon_crystal_batch(local_indices, global_indices_2d, f_mask, model, local)
        
        energy_ = np.ones((len(structures), 1))
        for i, structure in enumerate(structures):
        
            try:
                e = predict_energy(structure)
                energy_[i, 0] = e
            except:
                e = 10
                energy_[i, 0] = e
                

        for i, structure in enumerate(structures):
            e = energy_[i, 0]
            if e >= save_threshold[0] and e <= save_threshold[1]:
                chemical_formula = structure.composition.formula
                if args.data_path == os.path.join(current_folder, "data/carbon_24"):
                    key = str(time.time())
                    collection[f"{chemical_formula}_{key}"] = (structure, e)
                else:
                    if chemical_formula not in collection.keys():
                        collection[chemical_formula] = (structure, e)
                    else:
                        if collection[chemical_formula][1] > e:
                            collection[chemical_formula] = (structure, e)
        return energy_
    time1 = time.time()
    GA = ga(function=fitness_function if not batch else fitness_function_batch,\
            dimension=dimension,\
            variable_type='int',\
            variable_boundaries=varbound,\
            algorithm_parameters=GA_params,
            convergence_curve=False,
            progress_bar=True,
            init_population = init_population if init else None,
            batch = batch)
    GA.run()
    if len(collection.keys()) > 0 and save:
        sample_folder = os.path.join(temp_folder, f"denovo/{args.machine_id}_{args.base_id}_{timestamp_str}/sample")
        os.makedirs(sample_folder, exist_ok=True)
        collection_list = collection.items()
        if args.data_path == os.path.join(temp_folder, "data/carbon_24"):
            collection_list = random.sample(collection_list, 50)
        for chemical_formula, (structure, energy) in collection_list:
            cif_writer = CifWriter(structure)
            cif_writer.write_file(os.path.join(sample_folder, f"{energy:.3f}_{chemical_formula}.cif"))
        time2 = time.time()
    if post_optimize:
        save_folder = os.path.join(temp_folder, f"denovo/{args.machine_id}_{args.base_id}_{timestamp_str}/optimize")
        os.makedirs(save_folder, exist_ok=True)
        num_crystal = optimize_crystals(sample_folder, save_folder, save_threshold, check_convergence=check_convergence)
        # shutil.rmtree(sample_folder)
    time3 = time.time()
    print("################################################")
    print(f"GA strcutures: {len(collection.keys())}")
    print(f"GA time: {time2-time1:.2f}s")
    print(f"Num solutions: {num_crystal}")
    print(f"Optimize time: {time3-time2:.2f}s")
    print("################################################")

if __name__ == "__main__":
    with open(args.config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    dimension = config["num_quantizers2"]
    varbound=np.array([[0, config["codebooksize2"]-1]] * dimension)
    algorithm_param = {'max_num_iteration': 10,\
                   'population_size':64,\
                   'mutation_probability':0.4,\
                   'elit_ratio': 0.02,\
                   'crossover_probability': 0.5,\
                   'parents_portion': 0.2,\
                   'crossover_type':'uniform',\
                   'max_iteration_without_improv':None}
    seed = None
    # if seed is not None:
    #     set_seed(seed)
    # else:
    #     set_seed(int(time.time() % 1000))
    if args.base_id < 0:
        args.base_id = random.randint(0, 18928-1)
    GA_sample(base=args.base_id, keep = 100, base_size = 1000, dimension=dimension, varbound=varbound,
                            GA_params=algorithm_param, init=False, save=True, save_threshold=[-200, -20], seed = None,
                            batch = True, local=True, post_optimize=True, check_convergence=False)

