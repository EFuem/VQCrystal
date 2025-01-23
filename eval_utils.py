import itertools
import numpy as np
import torch
import hydra
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader
import random
from torch_geometric.data import Data
from scipy.spatial.distance import pdist
from scipy.spatial.distance import cdist
from hydra.experimental import compose
from hydra import initialize_config_dir
from pathlib import Path
from pymatgen.io.cif import CifParser
import smact
from smact.screening import pauling_test
from pymatgen.analysis import local_env
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from p_tqdm import p_umap
CrystalNN = local_env.CrystalNN(
    distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)
element_to_atomic_number = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40,
    'Nb': 41, 'Mo': 42, 'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58, 'Pr': 59, 'Nd': 60,
    'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66, 'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70,
    'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Tl': 81, 'Pb': 82, 'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96, 'Bk': 97, 'Cf': 98, 'Es': 99, 'Fm': 100,
    'Md': 101, 'No': 102, 'Lr': 103, 'Rf': 104, 'Db': 105, 'Sg': 106, 'Bh': 107, 'Hs': 108, 'Mt': 109,
    'Ds': 110, 'Rg': 111, 'Cn': 112, 'Nh': 113, 'Fl': 114, 'Mc': 115, 'Lv': 116, 'Ts': 117, 'Og': 118
}
chemical_symbols = [
    # 0
    'X',
    # 1
    'H', 'He',
    # 2
    'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
    # 3
    'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
    # 4
    'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
    # 5
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
    # 6
    'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
    'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
    'Po', 'At', 'Rn',
    # 7
    'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']

CompScalerMeans = [
    21.194441759304013,
    58.20212663122281,
    37.0076848719188,
    36.52738520455582,
    13.350626389725019,
    29.468922184630255,
    28.71735137747704,
    78.8868535524408,
    50.16950217496375,
    59.56764743604155,
    19.020429484306277,
    61.335572740454325,
    47.14515893344343,
    141.75135923307818,
    94.60620029962553,
    85.95794070476977,
    34.07300576173523,
    68.06189371516912,
    637.9862061297893,
    1817.2394155466848,
    1179.2532094169414,
    1127.2743149568837,
    431.51034284549826,
    909.1060025135899,
    3.7744320927984534,
    13.673707104881585,
    9.899275012083132,
    9.620186927095652,
    3.8426065581251856,
    9.96950217496375,
    3.305461575640406,
    5.483035282745288,
    2.1775737071048815,
    4.215114560306594,
    0.8206087101824266,
    3.732092798453359,
    109.16732721121315,
    179.5570323827936,
    70.38970517158047,
    136.0978305229613,
    27.027545809538527,
    119.16713388110198,
    1.2721433060967857,
    2.4614001837260617,
    1.1892568776289631,
    1.9844483610247092,
    0.4691462290494881,
    2.100143582306204,
    1.4829869502174964,
    1.9899951667472209,
    0.5070082165297245,
    1.7956250375970633,
    0.2056251946617602,
    1.745867568873852,
    0.05650072498791687,
    2.3618656355727405,
    2.3053649105848235,
    1.2829636137262992,
    0.9995555685850794,
    1.5150314161430642,
    0.7731271145480909,
    7.4648139197680035,
    6.691686805219913,
    4.010677272036105,
    2.612307566507693,
    3.303528274528758,
    0.2739487675205413,
    5.889753504108265,
    5.615804736587724,
    2.3244356612494683,
    2.1426251769710905,
    1.4464475592073465,
    4.739246012566457,
    14.578395360077332,
    9.839149347510874,
    9.413701584608935,
    3.537059747455868,
    8.550410826486225,
    0.008119864668922184,
    0.43286611889801835,
    0.4247462542290962,
    0.16687837041055423,
    0.17139889490813626,
    0.10898985016916385,
    0.06283228612856452,
    2.6573707104881583,
    2.594538424359594,
    1.219602938224228,
    1.0596390454742999,
    1.1120831319478008,
    0.14842919284678588,
    3.8473658772353794,
    3.6989366843885936,
    1.4541605082183982,
    1.3862277372859781,
    0.8018849685838569,
    0.03542774287095215,
    2.4474625422909617,
    2.4120347994200095,
    0.7745217539010397,
    0.9145812330586208,
    0.3198646689221846,
    1.552730787820203,
    6.910681488641856,
    5.357950700821653,
    3.615163570754227,
    1.9072256165179793,
    2.6702271628806185,
    14.608536589568727,
    34.83222477045747,
    20.223688180890715,
    22.47901710732293,
    7.17674504190757,
    18.641837024143584,
    0.009066988883518605,
    0.9185191396809959,
    0.9094521507974755,
    0.4368550481994018,
    0.38905942883427047,
    0.48375558240695804,
    0.0012985909686158003,
    0.21708593995837092,
    0.21578734898975546,
    0.08167977375391729,
    0.08155386250705281,
    0.06036340747305611,
    116.32010633156113,
    217.5905751570807,
    101.27046882551957,
    162.87154200548844,
    41.920624308665566,
    136.4664572257129]

CompScalerStds = [
    16.35781741152948,
    20.189540126474725,
    20.516298414514758,
    16.816765336550194,
    7.966591328222124,
    22.270791076753067,
    21.802116630115243,
    12.804546460581966,
    24.756629388687983,
    13.930306216047477,
    10.214535652334533,
    27.801612936980938,
    39.74031558353379,
    54.269739685575814,
    53.70466607591569,
    42.852342044453444,
    20.78341194242935,
    56.28783510219931,
    563.8004405882157,
    732.0722574247563,
    736.2122907972664,
    606.351603075103,
    272.62646060896407,
    810.6156779688841,
    3.0362262146833428,
    3.2075174256751606,
    4.0633818989245665,
    2.9738244769894764,
    1.7805586029644034,
    5.643243225066782,
    1.1994336274579853,
    0.8939013979423364,
    1.2297581799896975,
    1.0066021334519983,
    0.49129747526397105,
    1.4159553146070951,
    31.754756468836774,
    28.054241463256226,
    38.16336054795611,
    25.83485338379922,
    15.388376641904662,
    39.67137484594156,
    0.31988340032011076,
    0.6833658037760536,
    0.7464197945553585,
    0.4881349085029781,
    0.3176591553643101,
    0.8601748146737138,
    0.5864801661863596,
    0.10048913710210677,
    0.5836289120986499,
    0.2811748167435902,
    0.2468696279341553,
    0.5007375747433073,
    0.37237566669029587,
    1.7235989187720187,
    1.7058836077743305,
    1.1558859351244697,
    0.7677842566598179,
    1.9203550253462733,
    2.1289400248865182,
    3.5326064169848332,
    3.708508303762512,
    2.8709941136664567,
    1.6110681295257014,
    4.310192504023775,
    1.6644182118209292,
    6.228287671164213,
    6.1200848808512305,
    3.1986202996110302,
    2.4492978142248867,
    4.030497343977163,
    3.662028270049814,
    6.8192125550358345,
    6.614243783887738,
    4.334987449618594,
    2.568319610320196,
    5.9494890200106925,
    0.08974370432893491,
    0.4954725441517777,
    0.494304434278516,
    0.2309340434963803,
    0.2072873961103969,
    0.31162647950590266,
    0.39805702757060923,
    1.8111691089355726,
    1.7973395144505941,
    0.9486995373104102,
    0.7538753151875139,
    1.5233177017753785,
    0.7952606701778913,
    3.711190225170556,
    3.638721437232604,
    1.7171165424006831,
    1.4307904413917036,
    2.1047820817622904,
    0.49193748323158065,
    4.064840532426175,
    4.035286619587313,
    1.4858577214526643,
    1.5799117659864677,
    1.6130080156145745,
    1.555249156140194,
    4.776932951077492,
    4.569790780459629,
    2.224617778217326,
    1.7217507416156546,
    2.5969733650703763,
    7.215001918238936,
    19.252513469778584,
    18.775394044177858,
    9.447222764774764,
    6.7467931836261235,
    11.106825644766616,
    0.27206794253092115,
    1.6449321034573106,
    1.6236282792648686,
    0.8506917026741503,
    0.7020945355184042,
    1.2281895279350408,
    0.04134438177238229,
    0.5508855867341717,
    0.5486095551438679,
    0.24239297524046477,
    0.2127779137935831,
    0.3036750942874694,
    80.06063945615361,
    21.345794811194104,
    80.16475677581042,
    52.58533928558554,
    35.40836791039412,
    85.980205895116]

def smact_validity(comp, count,
                   use_pauling_test=True,
                   include_alloys=True):
    elem_symbols = tuple([chemical_symbols[elem] for elem in comp])
    space = smact.element_dictionary(elem_symbols)
    smact_elems = [e[1] for e in space.items()]
    electronegs = [e.pauling_eneg for e in smact_elems]
    ox_combos = [e.oxidation_states for e in smact_elems]
    if len(set(elem_symbols)) == 1:
        return True
    if include_alloys:
        is_metal_list = [elem_s in smact.metals for elem_s in elem_symbols]
        if all(is_metal_list):
            return True

    threshold = np.max(count)
    compositions = []
    for ox_states in itertools.product(*ox_combos):
        stoichs = [(c,) for c in count]
        # Test for charge balance
        cn_e, cn_r = smact.neutral_ratios(
            ox_states, stoichs=stoichs, threshold=threshold)
        # Electronegativity test
        if cn_e:
            if use_pauling_test:
                try:
                    electroneg_OK = pauling_test(ox_states, electronegs)
                except TypeError:
                    # if no electronegativity data, assume it is okay
                    electroneg_OK = True
            else:
                electroneg_OK = True
            if electroneg_OK:
                for ratio in cn_r:
                    compositions.append(
                        tuple([elem_symbols, ox_states, ratio]))
    compositions = [(i[0], i[2]) for i in compositions]
    compositions = list(set(compositions))
    if len(compositions) > 0:
        return True
    else:
        return False


def structure_validity(crystal, cutoff=0.5):
    dist_mat = crystal.distance_matrix
    # Pad diagonal with a large number
    dist_mat = dist_mat + np.diag(
        np.ones(dist_mat.shape[0]) * (cutoff + 10.))
    if dist_mat.min() < cutoff or crystal.volume < 0.1:
        return False
    else:
        return True
def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)
   
def get_cif_data(cif_path):
    try:
        # 直接从CIF文件手动解析信息
        with open(cif_path, 'r') as file:
            lines = file.readlines()
        frac_coords, atom_types, lengths, angles = parse_cif_file(lines)
    except Exception as e:
        print(f"Error parsing {cif_path} manually: {e}")
        return None
    
    cif_data = {
        'frac_coords': frac_coords,
        'atom_types': atom_types,
        'lengths': lengths,
        'angles': angles
    }
    return cif_data

def parse_cif_file(lines):
    frac_coords = []
    atom_types = []
    lengths = []
    angles = []

    atom_site_section = False
    for line in lines:
        if "_cell_length_a" in line:
            lengths.append(float(line.split()[-1]))
        elif "_cell_length_b" in line:
            lengths.append(float(line.split()[-1]))
        elif "_cell_length_c" in line:
            lengths.append(float(line.split()[-1]))
        elif "_cell_angle_alpha" in line:
            angles.append(float(line.split()[-1]))
        elif "_cell_angle_beta" in line:
            angles.append(float(line.split()[-1]))
        elif "_cell_angle_gamma" in line:
            angles.append(float(line.split()[-1]))
        elif "_atom_site_occupancy" in line or "_atom_site_fract_z" in line:
            atom_site_section = True
            continue
        elif atom_site_section and line.strip():
            parts = line.split()
            atom_symbol = parts[0]
            atom_number = element_to_atomic_number.get(atom_symbol, -1)  # 获取原子序号
            if atom_number == -1:
                raise ValueError(f"元素符号 {atom_symbol} 无效或不在字典中")
            atom_types.append(atom_number)
            frac_coords.append([float(parts[3]), float(parts[4]), float(parts[5])])

    return frac_coords, atom_types, lengths, angles

class StandardScaler:
    """A :class:`StandardScaler` normalizes the features of a dataset.
    When it is fit on a dataset, the :class:`StandardScaler` learns the
        mean and standard deviation across the 0th axis.
    When transforming a dataset, the :class:`StandardScaler` subtracts the
        means and divides by the standard deviations.
    """

    def __init__(self, means=None, stds=None, replace_nan_token=None):
        """
        :param means: An optional 1D numpy array of precomputed means.
        :param stds: An optional 1D numpy array of precomputed standard deviations.
        :param replace_nan_token: A token to use to replace NaN entries in the features.
        """
        self.means = means
        self.stds = stds
        self.replace_nan_token = replace_nan_token

    def fit(self, X):
        """
        Learns means and standard deviations across the 0th axis of the data :code:`X`.
        :param X: A list of lists of floats (or None).
        :return: The fitted :class:`StandardScaler` (self).
        """
        X = np.array(X).astype(float)
        self.means = np.nanmean(X, axis=0)
        self.stds = np.nanstd(X, axis=0)
        self.means = np.where(np.isnan(self.means),
                              np.zeros(self.means.shape), self.means)
        self.stds = np.where(np.isnan(self.stds),
                             np.ones(self.stds.shape), self.stds)
        self.stds = np.where(self.stds == 0, np.ones(
            self.stds.shape), self.stds)

        return self

    def transform(self, X):
        """
        Transforms the data by subtracting the means and dividing by the standard deviations.
        :param X: A list of lists of floats (or None).
        :return: The transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = (X - self.means) / self.stds
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none

    def inverse_transform(self, X):
        """
        Performs the inverse transformation by multiplying by the standard deviations and adding the means.
        :param X: A list of lists of floats.
        :return: The inverse transformed data with NaNs replaced by :code:`self.replace_nan_token`.
        """
        X = np.array(X).astype(float)
        transformed_with_nan = X * self.stds + self.means
        transformed_with_none = np.where(
            np.isnan(transformed_with_nan), self.replace_nan_token, transformed_with_nan)

        return transformed_with_none
    
def get_fp_pdist(fp_array):
    if isinstance(fp_array, list):
        fp_array = np.array(fp_array)
    fp_pdists = pdist(fp_array)
    return fp_pdists.mean()

CompScaler = StandardScaler(
    means=np.array(CompScalerMeans),
    stds=np.array(CompScalerStds),
    replace_nan_token=0.)

def filter_fps(struc_fps, comp_fps):
    assert len(struc_fps) == len(comp_fps)

    filtered_struc_fps, filtered_comp_fps = [], []

    for struc_fp, comp_fp in zip(struc_fps, comp_fps):
        if struc_fp is not None and comp_fp is not None:
            filtered_struc_fps.append(struc_fp)
            filtered_comp_fps.append(comp_fp)
    return filtered_struc_fps, filtered_comp_fps

def compute_cov(crys, gt_crys,
                struc_cutoff, comp_cutoff, num_gen_crystals=None):
    struc_fps = [c.struct_fp for c in crys]
    comp_fps = [c.comp_fp for c in crys]
    gt_struc_fps = [c.struct_fp for c in gt_crys]
    gt_comp_fps = [c.comp_fp for c in gt_crys]

    assert len(struc_fps) == len(comp_fps)
    assert len(gt_struc_fps) == len(gt_comp_fps)

    # Use number of crystal before filtering to compute COV
    if num_gen_crystals is None:
        num_gen_crystals = len(struc_fps)

    struc_fps, comp_fps = filter_fps(struc_fps, comp_fps)

    comp_fps = CompScaler.transform(comp_fps)
    gt_comp_fps = CompScaler.transform(gt_comp_fps)

    struc_fps = np.array(struc_fps)
    gt_struc_fps = np.array(gt_struc_fps)
    comp_fps = np.array(comp_fps)
    gt_comp_fps = np.array(gt_comp_fps)

    struc_pdist = cdist(struc_fps, gt_struc_fps)
    comp_pdist = cdist(comp_fps, gt_comp_fps)

    struc_recall_dist = struc_pdist.min(axis=0)
    struc_precision_dist = struc_pdist.min(axis=1)
    comp_recall_dist = comp_pdist.min(axis=0)
    comp_precision_dist = comp_pdist.min(axis=1)

    cov_recall = np.mean(np.logical_and(
        struc_recall_dist <= struc_cutoff,
        comp_recall_dist <= comp_cutoff))
    cov_precision = np.sum(np.logical_and(
        struc_precision_dist <= struc_cutoff,
        comp_precision_dist <= comp_cutoff)) / num_gen_crystals

    metrics_dict = {
        'cov_recall': cov_recall,
        'cov_precision': cov_precision,
        'amsd_recall': np.mean(struc_recall_dist),
        'amsd_precision': np.mean(struc_precision_dist),
        'amcd_recall': np.mean(comp_recall_dist),
        'amcd_precision': np.mean(comp_precision_dist),
    }

    combined_dist_dict = {
        'struc_recall_dist': struc_recall_dist.tolist(),
        'struc_precision_dist': struc_precision_dist.tolist(),
        'comp_recall_dist': comp_recall_dist.tolist(),
        'comp_precision_dist': comp_precision_dist.tolist(),
    }

    return metrics_dict, combined_dist_dict

def abs_cap(val, max_abs_val=1):
    """
    Returns the value with its absolute value capped at max_abs_val.
    Particularly useful in passing values to trignometric functions where
    numerical errors may result in an argument > 1 being passed in.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/util/num.py#L15
    Args:
        val (float): Input value.
        max_abs_val (float): The maximum absolute value for val. Defaults to 1.
    Returns:
        val if abs(val) < 1 else sign of val * max_abs_val.
    """
    return max(min(val, max_abs_val), -max_abs_val)
def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    """Converts lattice from abc, angles to matrix.
    https://github.com/materialsproject/pymatgen/blob/b789d74639aa851d7e5ee427a765d9fd5a8d1079/pymatgen/core/lattice.py#L311
    """
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    # Sometimes rounding errors result in values slightly > 1.
    val = abs_cap(val)
    gamma_star = np.arccos(val)

    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


def build_crystal_graph(crystal, graph_method='crystalnn'):
    """
    """

    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms

def preprocess_tensors(crystal_array_list, niggli, primitive, graph_method):
    def process_one(batch_idx, crystal_array, niggli, primitive, graph_method):
        frac_coords = crystal_array['frac_coords']
        atom_types = crystal_array['atom_types']
        lengths = crystal_array['lengths']
        angles = crystal_array['angles']
        crystal = Structure(
            lattice=Lattice.from_parameters(
                *(list(lengths) + list(angles))),
            species=atom_types,
            coords=frac_coords,
            coords_are_cartesian=False)
        graph_arrays = build_crystal_graph(crystal, graph_method)
        result_dict = {
            'batch_idx': batch_idx,
            'graph_arrays': graph_arrays,
        }
        return result_dict

    unordered_results = p_umap(
        process_one,
        list(range(len(crystal_array_list))),
        crystal_array_list,
        [niggli] * len(crystal_array_list),
        [primitive] * len(crystal_array_list),
        [graph_method] * len(crystal_array_list),
        num_cpus=30,
    )
    ordered_results = list(
        sorted(unordered_results, key=lambda x: x['batch_idx']))
    return ordered_results
    
def add_scaled_lattice_prop(data_list, lattice_scale_method):
    for dict in data_list:
        graph_arrays = dict['graph_arrays']
        # the indexes are brittle if more objects are returned
        lengths = graph_arrays[2]
        angles = graph_arrays[3]
        num_atoms = graph_arrays[-1]
        assert lengths.shape[0] == angles.shape[0] == 3
        assert isinstance(num_atoms, int)

        if lattice_scale_method == 'scale_length':
            lengths = lengths / float(num_atoms)**(1/3)

        dict['scaled_lattice'] = np.concatenate([lengths, angles])
        
class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

def get_model_path(eval_model_name):
    model_path = Path('prop_models') / eval_model_name
    return model_path.resolve()

def load_config(model_path):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
    return cfg
def load_model(model_path, load_data=False, testing=True):
    with initialize_config_dir(str(model_path)):
        cfg = compose(config_name='hparams')
        model = hydra.utils.instantiate(
            cfg.model,
            optim=cfg.optim,
            data=cfg.data,
            logging=cfg.logging,
            _recursive_=False,
        )
        ckpts = list(model_path.glob('*.ckpt'))
        if len(ckpts) > 0:
            ckpt_epochs = np.array(
                [int(ckpt.parts[-1].split('-')[0].split('=')[1]) for ckpt in ckpts])
            ckpt = str(ckpts[ckpt_epochs.argsort()[-1]])
        model = model.load_from_checkpoint(ckpt)
        model.lattice_scaler = torch.load(model_path / 'lattice_scaler.pt')
        model.scaler = torch.load(model_path / 'prop_scaler.pt')

        if load_data:
            datamodule = hydra.utils.instantiate(
                cfg.data.datamodule, _recursive_=False, scaler_path=model_path
            )
            if testing:
                datamodule.setup('test')
                test_loader = datamodule.test_dataloader()[0]
            else:
                datamodule.setup()
                test_loader = datamodule.val_dataloader()[0]
        else:
            test_loader = None

    return model, test_loader, cfg

def prop_model_eval(eval_model_name, crystal_array_list):

    model_path = get_model_path(eval_model_name)

    model, _, _ = load_model(model_path)
    cfg = load_config(model_path)

    dataset = TensorCrystDataset(
        crystal_array_list, cfg.data.niggli, cfg.data.primitive,
        cfg.data.graph_method, cfg.data.preprocess_workers,
        cfg.data.lattice_scale_method)

    dataset.scaler = model.scaler.copy()

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=256,
        num_workers=0,
        worker_init_fn=worker_init_fn)

    model.eval()

    all_preds = []

    for batch in loader:
        preds = model(batch)
        model.scaler.match_device(preds)
        scaled_preds = model.scaler.inverse_transform(preds)
        all_preds.append(scaled_preds.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0).squeeze(1)
    return all_preds.tolist()
