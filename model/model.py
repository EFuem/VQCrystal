import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import torch.nn.functional as Func
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
from tools import similarity_matrix, max_pooling_matching
import pdb
import torch.nn.functional as F
from utils import *
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core.structure import Structure
import pymatgen
import json
from vector_quantize_pytorch import VectorQuantize, ResidualVQ, GroupedResidualVQ
from pymatgen.vis.structure_vtk import StructureVis
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure, Lattice
import itertools
from model.cspnet import CSPNet
from lam_optimize.main import relax_run
from lam_optimize.relaxer import Relaxer
from pathlib import Path
# # GCNConv
class GraphConv(nn.Module):

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super(GraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.ReLU(), nn.LayerNorm(out_dim)
        )

    def forward(self, x: Tensor, adj: Tensor):
        # calculate AX
        y = torch.matmul(adj, x)
        # calculate AXW
        y = self.linear(y)
        return y


class GraphAttention(nn.Module):
    def __init__(self, d_model):
        super(GraphAttention, self).__init__()
        self.d_model = d_model
        self.scale = 1 / (d_model**0.5)

    def forward(self, q, k, v, adj_matrix):
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores * adj_matrix
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class GraphTransformerLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = GraphAttention(d_model)
        self.linear1 = nn.Linear(d_model, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, adj_matrix):
        src2 = self.self_attn(src, src, src, adj_matrix)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        src2 = self.linear2(F.relu(self.linear1(src)))
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class GraphTransformer(nn.Module):
    def __init__(self, d_model, nhead, num_layers):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList(
            [GraphTransformerLayer(d_model, nhead) for _ in range(num_layers)]
        )

    def forward(self, src, adj_matrix):
        for layer in self.layers:
            src = layer(src, adj_matrix)
        return src


class Transformer(nn.Module):
    def __init__(
        self, d_model, nhead=4, dim_feedforward=384, dropout=0.2, num_layers=2
    ):
        super(Transformer, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerLayer(nn.Module):
    def __init__(self, d_model, nhead=4, dim_feedforward=256, dropout=0.1):
        super(TransformerLayer, self).__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.linear1(src)))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.transformer_layer_atom = GraphTransformer(
            atom_fea_len, nhead=8, num_layers=1
        )
        # self.fc_transformer_nbr=nn.Linear(atom_fea_len*2+nbr_fea_len,nearest_multiple(atom_fea_len*2+nbr_fea_len,8))
        # self.transformer_layer_nbr = Transformer(nearest_multiple(atom_fea_len*2+nbr_fea_len,8), num_layers=0)
        # self.fc_full = nn.Linear(nearest_multiple(atom_fea_len*2+nbr_fea_len,8),
        #                          2*self.atom_fea_len)
        self.fc_full = nn.Linear(atom_fea_len * 2 + nbr_fea_len, 2 * self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.ln1 = nn.LayerNorm(2 * self.atom_fea_len)
        self.ln2 = nn.LayerNorm(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, adj):
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # Separate atom_in_fea into batches
        atom_batches = [atom_in_fea[idx_map] for idx_map in crystal_atom_idx]
        # Compute lengths of each batch
        batch_lengths = [len(idx_map) for idx_map in crystal_atom_idx]
        # Pad each batch to same length for transformer
        atom_batches_padded = nn.utils.rnn.pad_sequence(atom_batches).permute(1, 0, 2)
        # Pass through transformer
        atom_batches_transformed = self.transformer_layer_atom(atom_batches_padded, adj)
        # print(atom_batches_transformed)
        # atom_batches_transformed=atom_batches_transformed.permute(1,0,2)
        atom_batches_transformed_list = [
            atom_batches_transformed[i, : batch_lengths[i], :]
            for i in range(len(batch_lengths))
        ]
        atom_in_fea = torch.cat(atom_batches_transformed_list)

        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                atom_nbr_fea,
                nbr_fea,
            ],
            dim=2,
        )
        # total_nbr_fea=self.fc_transformer_nbr(total_nbr_fea)
        # total_nbr_fea=self.transformer_layer_nbr(total_nbr_fea)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(N, M, self.atom_fea_len * 2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """

    def __init__(
        self,
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=64,
        n_conv=3,
        h_fea_len=128,
        n_h=1,
        classification=False,
    ):

        super(CrystalGraphConvNet, self).__init__()
        self.classification = classification
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.trans_atom = Transformer(h_fea_len, num_layers=2)
        if n_h > 1:
            self.fcs = nn.ModuleList(
                [nn.Linear(h_fea_len, h_fea_len) for _ in range(n_h - 1)]
            )
            self.softpluses = nn.ModuleList([nn.Softplus() for _ in range(n_h - 1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()

    def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, A):

        atom_fea = self.embedding(atom_fea)
        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx, A)

        max_len = max([len(idx_map) for idx_map in crystal_atom_idx])
        padded_fea = []
        for idx_map in crystal_atom_idx:
            # Select the features for the current crystal
            fea = atom_fea[idx_map]
            # Calculate the number of padding rows needed
            pad_len = max_len - fea.shape[0]
            # Create a tensor of zeros for padding
            pad = torch.zeros(pad_len, fea.shape[1], device=fea.device)
            # Concatenate the feature and padding tensors
            padded_fea.append(torch.cat([fea, pad], dim=0))
        padded_fea = torch.stack(padded_fea, dim=0)
        return padded_fea

    def pooling(self, atom_fea, crystal_atom_idx):

        assert (
            sum([len(idx_map) for idx_map in crystal_atom_idx])
            == atom_fea.data.shape[0]
        )
        summed_fea = [
            torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
            for idx_map in crystal_atom_idx
        ]
        return torch.cat(summed_fea, dim=0)


class GraphVAE(nn.Module):

    def __init__(
        self,
        loss=nn.CrossEntropyLoss(),
        class_dim: int = 100,
        pos_dim: int = 3,
        lattice_dim: int = 6,
        hidden_dim: int = 256,
        latent_dim: int = 32,
        k: int = 64,
        encoder1_dim=[128, 128],
        num_encoder2_layers: int = 3,
        decoder2_dims=[32],
        decoder1_dims=[64, 256],
        decoder_lattice_dims=[64, 64],
        decoder_lattice_dims2=[64, 64],
        decoder_prop_dims=[64, 64],
        decoder_force_dims=[64, 64],
        codebooksize1=32,
        codebooksize2=16,
        codebooksizel=16,
        num_quantizers1=4,
        num_quantizers2=4,
        num_quantizersl=4,
        sample_codebook_temp=0.1,
        lambdas=None,
        temperature: float = 0.8,
        commitment_cost: float = 0.25,
        normalizer=None,
        normalizer_properties=None,
    ) -> None:
        """init

        Args:
            input_dim (int): input feature dimension for node
            hidden_dim (int): hidden dim for 2-layer gcn
            latent_dim (int): dimension of the latent representation of graph
            k (int): max number of nodes
            pool (str, optional): pooling strategy. Defaults to "sum".
        """
        super().__init__()
        self.d1 = class_dim
        self.d2 = pos_dim
        self.d3 = lattice_dim
        self.input_dim = class_dim + pos_dim
        self.encoder1_dim = encoder1_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.k = k
        self.decoder_global_dims = decoder2_dims
        self.decoder_dims = decoder1_dims
        self.decoder_lattice_dims = decoder_lattice_dims
        self.decoder_lattice_dims2 = decoder_lattice_dims2
        self.decoder_prop_dims = decoder_prop_dims
        self.decoder_force_dims = decoder_force_dims
        self.epoch = 0
        self.step = 0
        # DWA
        if lambdas is None:
            self.lambdas = np.array([15, 100, 0.25, 50, 0.1])  # V, P,Comm,Pre
        else:
            self.lambdas = np.array(lambdas)
        self.normalizer = normalizer
        self.normalizer_properties=normalizer_properties
        self.matcher = StructureMatcher(
            ltol=0.3,
            stol=0.5,
            angle_tol=10,
            scale=True,
            attempt_supercell=True,
        )

        # pdb.set_trace()
        self.encoder_csp=CSPNet(hidden_dim=latent_dim,num_layers=3)
        self.proj_to_vq1 = nn.Linear(latent_dim * 2, latent_dim)
        self.proj_to_latent1 = nn.Linear(hidden_dim, latent_dim)
        self.proj_to_latent2 = nn.Linear(hidden_dim, latent_dim)
        self.encoderl = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Dropout(0.1),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.encoder1_dim.insert(0, self.input_dim + 6)
        self.encoder1 = nn.ModuleList()
        for i in range(len(self.encoder1_dim) - 1):
            self.encoder1.append(
                nn.Sequential(
                    nn.Linear(self.encoder1_dim[i], self.encoder1_dim[i + 1]),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.encoder1.append(
                nn.TransformerEncoderLayer(
                    self.encoder1_dim[i + 1], 4, 256, batch_first=True
                )
            )
        self.encoder1.append(
            nn.Sequential(
                nn.Linear(self.encoder1_dim[-1], hidden_dim), nn.LeakyReLU(0.1)
            )
        )
        self.encoder2 = CrystalGraphConvNet(
            orig_atom_fea_len=self.hidden_dim,
            nbr_fea_len=253,
            n_conv=num_encoder2_layers,
            h_fea_len=hidden_dim,
            atom_fea_len=hidden_dim,
        )

        self.vq1 = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers1,
            codebook_size=codebooksize1,
            stochastic_sample_codes=True,
            sample_codebook_temp=sample_codebook_temp,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook=True,  # whether to share the codebooks for all quantizers or not
            kmeans_init=True,  # set to True
            kmeans_iters=10,
        )
        self.vq2 = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers2,
            codebook_size=codebooksize2,
            stochastic_sample_codes=True,
            sample_codebook_temp=sample_codebook_temp,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook=True,  # whether to share the codebooks for all quantizers or not
            kmeans_init=True,  # set to True
            kmeans_iters=10,
        )
        self.vql = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizersl,
            codebook_size=codebooksizel,
            stochastic_sample_codes=True,
            sample_codebook_temp=sample_codebook_temp,  # temperature for stochastically sampling codes, 0 would be equivalent to non-stochastic
            shared_codebook=True,  # whether to share the codebooks for all quantizers or not
            kmeans_init=True,  # set to True
            kmeans_iters=10,
        )

        self.decoder_dims.insert(0, self.latent_dim)
        self.decoder = nn.ModuleList()
        for i in range(len(self.decoder_dims) - 1):
            self.decoder.append(
                nn.Sequential(
                    nn.Linear(self.decoder_dims[i], self.decoder_dims[i + 1]),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder.append(
                nn.TransformerEncoderLayer(
                    self.decoder_dims[i + 1], 4, 256, batch_first=True
                )
            )
        self.decoder.append(
            nn.Sequential(nn.Linear(self.decoder_dims[-1], self.input_dim))
        )
        self.decode_global_bias = nn.Parameter(
            torch.zeros([self.k, latent_dim]), requires_grad=True
        )
        self.decoder_global_dims.insert(0, self.latent_dim)
        self.decoder_from_global = nn.ModuleList()
        for i in range(len(self.decoder_global_dims) - 1):
            self.decoder_from_global.append(
                nn.Sequential(
                    nn.Linear(
                        self.decoder_global_dims[i], self.decoder_global_dims[i + 1]
                    ),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder_from_global.append(
                nn.TransformerEncoderLayer(
                    self.decoder_global_dims[i + 1], 4, 256, batch_first=True
                )
            )
        self.decoder_from_global.append(
            nn.Sequential(nn.Linear(self.decoder_global_dims[-1], self.input_dim))
        )

        self.decoder_lattice_dims.insert(0, self.input_dim)
        self.decoder_lattice = nn.ModuleList()
        for i in range(len(self.decoder_lattice_dims) - 1):
            self.decoder_lattice.append(
                nn.Sequential(
                    nn.Linear(
                        self.decoder_lattice_dims[i], self.decoder_lattice_dims[i + 1]
                    ),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder_lattice.append(
                nn.TransformerEncoderLayer(
                    self.decoder_lattice_dims[i + 1], 4, 256, batch_first=True
                )
            )
        self.decoder_lattice.append(
            nn.Sequential(nn.Linear(self.decoder_lattice_dims[-1], 6))
        )


        #self.decoder_diff=CSPNet(hidden_dim=latent_dim,num_layers=2)
        self.decoder_prop = nn.ModuleList()
        self.decoder_prop_dims.insert(0, latent_dim*2)
        for i in range(len(self.decoder_prop_dims) - 1):
            self.decoder_prop.append(
                nn.Sequential(
                    nn.Linear(self.decoder_prop_dims[i], self.decoder_prop_dims[i + 1]),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder_prop.append(
                nn.TransformerEncoderLayer(
                    self.decoder_prop_dims[i + 1], 4, 256, batch_first=True
                )
            )
        self.decoder_prop.append(
            nn.Sequential(nn.Linear(self.decoder_prop_dims[-1], 1))
        )
        
        self.decoder_force_dims.insert(0, self.latent_dim)
        self.decoder_force = nn.ModuleList()
        for i in range(len(self.decoder_force_dims) - 1):
            self.decoder_force.append(
                nn.Sequential(
                    nn.Linear(
                        self.decoder_force_dims[i], self.decoder_force_dims[i + 1]
                    ),
                    nn.Dropout(0.1),
                    nn.LeakyReLU(0.1),
                )
            )
            self.decoder_force.append(
                nn.TransformerEncoderLayer(
                    self.decoder_force_dims[i + 1], 4, 256, batch_first=True
                )
            )
        self.decoder_force.append(
            nn.Sequential(nn.Linear(self.decoder_force_dims[-1], 3))
        )

        
        self.loss = loss
        self.commitment_cost = commitment_cost

    def encode(
        self,
        F: Tensor,
        f_mask: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
        A: Tensor,
        lattice: Tensor,
    ):
        for i, net in enumerate(self.encoder1):
            if i == 0:
                z1 = net(
                    torch.cat(
                        [
                            F,
                            lattice.unsqueeze(1).expand(
                                F.shape[0], F.shape[1], lattice.shape[-1]
                            ),
                        ],
                        dim=-1,
                    )
                )
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    z1 = net(z1, src_key_padding_mask=~f_mask)
                else:
                    z1 = net(z1)
                # pdb.set_trace()
        z1 = z1.masked_fill(~f_mask.unsqueeze(-1), 0)

        masked_z1 = z1[f_mask]
        reshaped_z1 = masked_z1.view(-1, z1.size(2))
        z2 = self.encoder2(reshaped_z1, nbr_fea, nbr_fea_idx, crystal_atom_idx, A)
        # pdb.set_trace()
        z2_masked = z2 * f_mask.unsqueeze(-1)
        z2_sum = torch.sum(z2_masked, dim=1)
        sequence_lengths = torch.sum(f_mask.float(), dim=1)
        z2 = z2_sum / sequence_lengths.unsqueeze(-1)  # (B, D)
        z1 = self.proj_to_latent1(z1)
        z1 = z1.masked_fill(~f_mask.unsqueeze(-1), 0)
        z2 = self.proj_to_latent2(z2)
        frac_coords_input = torch.clamp((F[..., self.d1 :][f_mask] + 1) / 2, 0, 1)
        num_atoms_input = f_mask.sum(dim=-1)
        node2batch_input = torch.repeat_interleave(
            torch.arange(num_atoms_input.size(0)).to(num_atoms_input.device),
            num_atoms_input,
        )
        lattice_matrix=batch_lattice_to_matrix_torch(lattice)
        atom_input = F[..., : self.d1].argmax(dim=-1)[f_mask] + 1
        local_feature, global_feature = self.encoder_csp(
            atom_input, frac_coords_input, lattice_matrix, num_atoms_input, node2batch_input
        )
        # local_feature=batch_crystals(local_feature,f_mask.sum(-1))
        # z1=z1+local_feature
        z2=z2+global_feature+z1.mean(dim=1)
        return (z1, z2)

    def quantize(self, z_e, f_mask):
        # pdb.set_trace()
        z1, z2 = z_e

        zq2, indices2, commitment_loss2 = self.vq2(z2.unsqueeze(1))
        zq2 = zq2.squeeze(1)
        zq1, indices1, commitment_loss1 = self.vq1(z1)
        return (
            (indices1, indices2),
            (zq1, zq2),
            (commitment_loss1, commitment_loss2),
        )

    # def graph_offset(self, F: Tensor, f_mask: Tensor, lattice: Tensor):
    #     if lattice.shape[-1]!=3:
    #         lattice=batch_lattice_to_matrix_torch(lattice).to(F.device)
    #     atom_input = F[..., : self.d1].argmax(dim=-1)[f_mask] + 1
    #     frac_coords_input = torch.clamp((F[..., self.d1 :][f_mask] + 1) / 2, 0, 1)
    #     num_atoms_input = f_mask.sum(dim=-1)
    #     node2batch_input = torch.repeat_interleave(
    #         torch.arange(num_atoms_input.size(0)).to(num_atoms_input.device),
    #         num_atoms_input,
    #     )
    #     lattice_out, coord_out = self.decoder_diff.forward_decode(
    #         atom_input, frac_coords_input, lattice, num_atoms_input, node2batch_input
    #     )
    #     lattice_out=matrix_to_lattice_params(lattice_out)
    #     F_new=F.clone()
    #     F_new = torch.cat([F_new[..., :-3], coord_out], dim=-1)
        
    #     return lattice_out,F_new
    
    def decode(self, z_e, z_q, f_mask):
        # unpack inputs
        zq1, zq2 = z_q
        # pdb.set_trace()
        if z_e is not None:
            z1, z2 = z_e
            # stop decoder optimization from accessing the embedding
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()

        # upsample quantized2 to match spacial dim of quantized1
        zq2_upsampled = zq2.unsqueeze(1).expand_as(zq1)
        # decode
        # combined_latents = torch.cat([zq1, zq2_upsampled], -1)
        combined_latents = zq1 + zq2_upsampled
        for i, net in enumerate(self.decoder):
            if i == 0:
                dec1_out = net(combined_latents)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec1_out = net(dec1_out, src_key_padding_mask=~f_mask)
                else:
                    dec1_out = net(dec1_out)
        dec1_out = dec1_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec1_out

    def decode_from_global(self, z2, zq2, zq1, f_mask):
        zq2 = z2 + (zq2 - z2).detach()
        zq2 = zq2.unsqueeze(1).expand_as(zq1)
        zq2 = zq2 + self.decode_global_bias.unsqueeze(0)[:, : zq2.shape[1], :]
        for i, net in enumerate(self.decoder_from_global):
            if i == 0:
                dec_out = net(zq2)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec_out = net(dec_out, src_key_padding_mask=~f_mask)
                else:
                    dec_out = net(dec_out)
        dec_out = dec_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec_out

    def decode_local(self, z_e, z_q, f_mask):
        # unpack inputs
        zq1, zq2 = z_q
        # pdb.set_trace()
        if z_e is not None:
            z1, z2 = z_e
            # stop decoder optimization from accessing the embedding
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()
        # upsample quantized2 to match spacial dim of quantized1
        zq2_upsampled = zq2.unsqueeze(1).expand_as(zq1)
        # decode
        combined_latents = zq1+zq2_upsampled
        for i, net in enumerate(self.decoder):
            if i == 0:
                dec1_out = net(combined_latents)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec1_out = net(dec1_out, src_key_padding_mask=~f_mask)
                else:
                    dec1_out = net(dec1_out)
        dec1_out = dec1_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec1_out

    def decode_lattice_from_recon(self, recon, f_mask):
        for i, net in enumerate(self.decoder_lattice):
            if i == 0:
                dec_out = net(recon)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec_out = net(dec_out, src_key_padding_mask=~f_mask)
                else:
                    dec_out = net(dec_out)
        dec_out = dec_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec_out
    
    def decode_force(self, z_e,z_q, f_mask):
        zq1, zq2 = z_q
        # pdb.set_trace()
        if z_e is not None:
            z1, z2 = z_e
            # stop decoder optimization from accessing the embedding
            zq1 = z1 + (zq1 - z1).detach()
            zq2 = z2 + (zq2 - z2).detach()
        # upsample quantized2 to match spacial dim of quantized1
        zq2_upsampled = zq2.unsqueeze(1).expand_as(zq1)
        # decode
        combined_latents = zq1+zq2_upsampled
        for i, net in enumerate(self.decoder_force):
            if i == 0:
                dec_out = net(combined_latents)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec_out = net(dec_out, src_key_padding_mask=~f_mask)
                else:
                    dec_out = net(dec_out)
        dec_out = dec_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec_out

    # def decode_lattice_from_vq(self, z_q, f_mask):
    #     (zq1, zq2) = z_q
    #     # pdb.set_trace()
    #     recon = torch.cat([zq1, zq2.unsqueeze(1).expand_as(zq1)], dim=-1)
    #     for i, net in enumerate(self.decoder_lattice2):
    #         if i == 0:
    #             dec_out = net(recon)
    #         else:
    #             if isinstance(net, nn.TransformerEncoderLayer):
    #                 dec_out = net(dec_out, src_key_padding_mask=~f_mask)
    #             else:
    #                 dec_out = net(dec_out)
    #     dec_out = dec_out.masked_fill(~f_mask.unsqueeze(-1), 0)
    #     return dec_out

    def decode_property_from_vq(self, z_q, f_mask):
        (zq1, zq2) = z_q
        # pdb.set_trace()
        recon = torch.cat([zq1, zq2.unsqueeze(1).expand_as(zq1)], dim=-1)
        for i, net in enumerate(self.decoder_prop):
            if i == 0:
                dec_out = net(recon)
            else:
                if isinstance(net, nn.TransformerEncoderLayer):
                    dec_out = net(dec_out, src_key_padding_mask=~f_mask)
                else:
                    dec_out = net(dec_out)
        dec_out = dec_out.masked_fill(~f_mask.unsqueeze(-1), 0)
        return dec_out

    def forward(
        self,
        F: Tensor,
        A: Tensor,
        f_mask: Tensor,
        lattice: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
        properties: Tensor,
        forces:Tensor,
    ):
        lattice_target = self.normalizer.norm(lattice)
        z_e = self.encode(
            F, f_mask, nbr_fea, nbr_fea_idx, crystal_atom_idx, A, lattice_target
        )
        # pdb.set_trace()
        indices, z_q, commitment_loss = self.quantize(z_e, f_mask)
        zq2 = z_q[1]
        recon_x = self.decode(z_e, z_q, f_mask)
        recon_from_global = self.decode_from_global(z_e[1], zq2, z_q[0], f_mask)
        recon_lattice = self.decode_lattice_from_recon(recon_x, f_mask).sum(dim=-2)
        prop_predict = self.decode_property_from_vq(z_q, f_mask).sum(dim=-2)
        #forces_predict=self.decode_force(z_e, z_q,f_mask)[f_mask]
        lattice_loss = (
            Func.mse_loss(recon_lattice, lattice_target)
            #+ Func.mse_loss(recon_lattice2, lattice_target)
        )

        V_loss, P_loss = self.reconstruction_loss(
            recon_x , F, f_mask
        )
        commitment_loss = sum([loss.sum() for loss in commitment_loss])
        properties=self.normalizer_properties.norm(properties)
        property_loss = Func.mse_loss(prop_predict, properties)
        #force_loss = Func.mse_loss(forces_predict, forces)
        return {
            'total_loss': self.lambdas[0] * V_loss 
                        + self.lambdas[1] * P_loss
                        + self.lambdas[2] * commitment_loss
                        + self.lambdas[3] * lattice_loss
                        + self.lambdas[4] * property_loss,
            'V_loss': V_loss,
            'P_loss': P_loss,
            'commitment_loss': commitment_loss,
            'lattice_loss': lattice_loss,
            'property_loss': property_loss
        }
        # return {
        #     'total_loss': self.lambdas[0] * (V_loss+V_loss_from_global) 
        #                 + self.lambdas[1] * (P_loss+P_loss_from_global) 
        #                 + self.lambdas[2] * commitment_loss
        #                 + self.lambdas[3] * lattice_loss
        #                 + self.lambdas[4] * property_loss,
        #     'V_loss': V_loss+V_loss_from_global,
        #     'P_loss': P_loss+P_loss_from_global,
        #     'commitment_loss': commitment_loss,
        #     'lattice_loss': lattice_loss,
        #     'property_loss': property_loss
        # }

    def evaluation_forward(
        self,
        F: Tensor,
        A: Tensor,
        f_mask: Tensor,
        lattice: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
        properties,
        forces,
        crystals,
        accuracy=True,
        dp_relax=False,
        dp_path=None,
    ):
        with torch.no_grad():
            len_lattice = lattice[..., :3]
            angle_lattice = lattice[..., 3:]
            lattice_target = self.normalizer.norm(lattice)
            z_e = self.encode(
                F, f_mask, nbr_fea, nbr_fea_idx, crystal_atom_idx, A, lattice_target
            )
            # pdb.set_trace()
            indices, z_q, commitment_loss = self.quantize(z_e, f_mask)
            zq2 = z_q[1]
            recon_x = self.decode(z_e, z_q, f_mask)
            recon_from_global = self.decode_from_global(
                z_e[1], zq2 + torch.sum(z_q[0], dim=1, keepdim=False), z_q[0], f_mask
            )
            recon_lattice1 = self.decode_lattice_from_recon(recon_x, f_mask).sum(dim=-2)
            #recon_lattice2 = self.decode_lattice_from_vq(z_q, f_mask).sum(dim=-2)
            recon_lattice = recon_lattice1
            lattice_loss = Func.mse_loss(recon_lattice, lattice_target)
            prop_predict = self.decode_property_from_vq(z_q, f_mask).sum(dim=-2)
            prop_predict=self.normalizer_properties.denorm(prop_predict)
            property_loss = Func.mse_loss(prop_predict, properties)
            lattice_predict = self.normalizer.denorm(recon_lattice)
            #forces_predict=self.decode_force(z_e, z_q,f_mask)[f_mask]
            # pdb.set_trace()
            V_loss, P_loss = self.reconstruction_loss(
                recon_x , F, f_mask
            )
            # V_loss_from_global, P_loss_from_global = self.reconstruction_loss(
            #     recon_from_global, F, f_mask
            # )
            commitment_loss = sum([loss.sum() for loss in commitment_loss])
            #force_loss = Func.mse_loss(forces_predict, forces)
            atom_acc = None
            match_accuracy = None
            mean_rms_dist = None
            match_accuracy_with_lattice = None
            mean_rms_dist_with_lattice = None
            # pdb.set_trace()
            recon_x = recon_x 
            if accuracy:
                recon_x[..., : self.d1] = Func.softmax(recon_x[..., : self.d1], dim=-1)
                atom_acc = self.atom_accuracy(F, recon_x, f_mask)
                match_accuracy, mean_rms_dist = self.structure_matcher_wo_lattice(
                    recon_x, f_mask, crystals
                )
                match_accuracy_with_lattice, mean_rms_dist_with_lattice = self.structure_matcher_with_lattice(
                    recon_x, f_mask, crystals,lattice_predict,dp_relax=dp_relax,dp_path=dp_path
                )
                
        return {
            'total_loss': self.lambdas[0] * V_loss
                        + self.lambdas[1] * P_loss
                        + self.lambdas[2] * commitment_loss
                        + self.lambdas[3] * lattice_loss
                        + self.lambdas[4] * property_loss,
            'V_loss': V_loss,
            'P_loss': P_loss,
            'commitment_loss': commitment_loss,
            'lattice_loss': lattice_loss,
            'property_loss': property_loss,
            'atom_acc': atom_acc,
            'match_accuracy': match_accuracy,
            'mean_rms_dist': mean_rms_dist,
            'match_accuracy_with_lattice': match_accuracy_with_lattice,
            'mean_rms_dist_with_lattice': mean_rms_dist_with_lattice
        }

    def reconstruction_loss(self, recon, F, f_mask):
        V_recon = recon[..., : self.d1]
        P_recon = recon[..., self.d1 :]
        V_recon_mask = V_recon[f_mask]
        P_recon_mask = P_recon[f_mask]
        V = F[..., : self.d1]
        P = F[..., self.d1 :]
        V_mask = V[f_mask]
        P_mask = P[f_mask]
        V_loss = self.loss(V_recon_mask, V_mask.argmax(dim=-1))
        P_loss = Func.mse_loss(P_recon_mask, P_mask)

        return (V_loss, P_loss)

    def get_structure_from_recon(self, F_recon, f_mask, lattice_predict):
        B = F_recon.shape[0]
        F_recon_ = F_recon  # B, N, K
        _, N, D = F_recon_.shape
        F_recon_pad = F_recon_.masked_fill(~f_mask.unsqueeze(-1).expand(B, N, D), 0)
        V_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=0, length=self.d1)
        P_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=self.d1, length=self.d2)
        P_recon_pad = (P_recon_pad + 1) / 2
        P_recon_pad = crystal_position_constrain(P_recon_pad)
        element_recon_pad = argmax_with_custom_value(V_recon_pad)
        P_list = tensor_to_list_based_on_mask(P_recon_pad, f_mask)
        element_list = tensor_to_list_based_on_mask(element_recon_pad, f_mask)
        lattice_list = lattice_predict.cpu().numpy().tolist()
        # lattice = Lattice.from_parameters(*lattice)
        structures = []
        for atom, position, lattice in zip(element_list, P_list, lattice_list):
            lattice = Lattice.from_parameters(*lattice)
            structure = Structure(
                lattice, int_to_element(atom.tolist()), position.tolist()
            )
            structures.append(structure)
        return structures


    
    def structure_matcher_with_lattice(self, F_recon, f_mask, crystals, lattice_predict,dp_relax=False,dp_path=None):
        self.step += 1
        match = 0
        B = F_recon.shape[0]
        F_recon_ = F_recon  # B, N, K
        _, N, D = F_recon_.shape
        F_recon_pad = F_recon_.masked_fill(~f_mask.unsqueeze(-1).expand(B, N, D), 0)
        V_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=0, length=self.d1)
        P_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=self.d1, length=self.d2)
        P_recon_pad = (P_recon_pad + 1) / 2
        P_recon_pad = crystal_position_constrain(P_recon_pad)
        element_recon_pad = argmax_with_custom_value(V_recon_pad)
        P_list = tensor_to_list_based_on_mask(P_recon_pad, f_mask)
        element_list = tensor_to_list_based_on_mask(element_recon_pad, f_mask)
        lattice_list=lattice_predict.tolist()
        idex=0
        structure = None
        rms = []
        for atom, position, lattice, crystal in zip(
            element_list, P_list, lattice_list, crystals
        ):
            idex+=1
            try:
                lattice = Lattice.from_parameters(*lattice)
                structure = Structure(
                    lattice, int_to_element(atom.tolist()), position.tolist()
                )
                if dp_relax:
                    with torch.enable_grad():
                        cif_writer = CifWriter(structure)
                        cif_writer.write_file(f"temp\\recon{self.step}.cif")
                        cif_writer.write_file(f"output\\origin{convert_atom_list_to_huaxuehsi(int_to_element(atom.tolist()))}.cif")
                        if dp_path:
                            relaxer = Relaxer(Path(dp_path))
                        else:
                            relaxer = Relaxer("mace")
                        res_df = relax_run(
                            Path("temp"),
                            relaxer,
                            steps=100,
                            check_duplicate=False
                        )
                        clear_folder("temp")
                        structure=Structure.from_dict(res_df['final_structure'][0])
                        cif_writer = CifWriter(structure)
                        cif_writer.write_file(f"output\\relax{convert_atom_list_to_huaxuehsi(int_to_element(atom.tolist()))}.cif")
                # pdb.set_trace()
                is_match = int(self.matcher.fit(structure, crystal))
                match += is_match
                rms_dist = self.matcher.get_rms_dist(structure, crystal)
                rms_dist = None if rms_dist is None else rms_dist[0]
                rms.append(rms_dist)
            except BaseException as e:
                print(e)
        rms_dists = np.array(rms)
        if structure:
            data_to_save = {"structure_recon": str(structure)}
            with open("structures_recon.txt", "a") as file:
                file.write("recon:" + str(structure) + "\n")
                file.write("crystal:" + str(crystal) + "\n")
                file.write("rms:" + str(rms_dists) + "\n")
                file.write("\n\n\n----------------------\n\n")
            cif_writer = CifWriter(structure)
            cif_writer2 = CifWriter(crystal)
            cif_writer.write_file(f"visual\\recon{self.step}.cif")
            cif_writer2.write_file(f"visual\\target{self.step}.cif")

        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return match / B, mean_rms_dist
    
            
    
    def structure_matcher_wo_lattice(self, F_recon, f_mask, crystals):
        self.step += 1
        match = 0
        B = F_recon.shape[0]
        F_recon_ = F_recon  # B, N, K
        _, N, D = F_recon_.shape
        F_recon_pad = F_recon_.masked_fill(~f_mask.unsqueeze(-1).expand(B, N, D), 0)
        V_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=0, length=self.d1)
        P_recon_pad = torch.narrow(F_recon_pad, dim=-1, start=self.d1, length=self.d2)
        P_recon_pad = (P_recon_pad + 1) / 2
        P_recon_pad = crystal_position_constrain(P_recon_pad)
        element_recon_pad = argmax_with_custom_value(V_recon_pad)
        P_list = tensor_to_list_based_on_mask(P_recon_pad, f_mask)
        element_list = tensor_to_list_based_on_mask(element_recon_pad, f_mask)

        structure = None
        rms = []
        for atom, position, crystal in zip(element_list, P_list, crystals):
            try:
                structure = Structure(
                    crystal.lattice, int_to_element(atom.tolist()), position.tolist()
                )
                # pdb.set_trace()
                is_match = int(self.matcher.fit(structure, crystal))
                match += is_match
                rms_dist = self.matcher.get_rms_dist(structure, crystal)
                rms_dist = None if rms_dist is None else rms_dist[0]
                rms.append(rms_dist)
            except BaseException as e:
                print(e)
        rms_dists = np.array(rms)
        if structure:
            data_to_save = {"structure_recon": str(structure)}
            with open("structures_recon.txt", "a") as file:
                file.write("recon:" + str(structure) + "\n")
                file.write("crystal:" + str(crystal) + "\n")
                file.write("rms:" + str(rms_dists) + "\n")
                file.write("\n\n\n----------------------\n\n")
            cif_writer = CifWriter(structure)
            cif_writer2 = CifWriter(crystal)
            cif_writer.write_file(f"visual\\recon{self.step}.cif")
            cif_writer2.write_file(f"visual\\target{self.step}.cif")

        mean_rms_dist = rms_dists[rms_dists != None].mean()
        return match / B, mean_rms_dist

    def atom_accuracy(self, F: Tensor, F_recon: Tensor, f_mask: Tensor):
        B = F.shape[0]
        F_recon_ = F_recon
        _, N, D = F_recon_.shape
        F_masked = F[f_mask]
        F_recon_masked = F_recon[f_mask]
        V = F_masked[:, : self.d1]
        V_recon = F_recon_masked[:, : self.d1]
        V_ = V.view(-1, self.d1)
        V_recon_ = V_recon.view(-1, self.d1)
        truth = torch.argmax(V_, dim=-1)
        prediction = torch.argmax(V_recon_, dim=-1)
        return (prediction == truth).float().mean()

    def get_crysal_codebook(
        self,
        F: Tensor,
        A: Tensor,
        f_mask: Tensor,
        lattice: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
    ):
        lattice_target = self.normalizer.norm(lattice)
        z_e = self.encode(
            F, f_mask, nbr_fea, nbr_fea_idx, crystal_atom_idx, A, lattice_target
        )
        # pdb.set_trace()
        indices, z_q, commitment_loss = self.quantize(z_e, f_mask)
        return indices, z_e, z_q

    def predict_using_codebook(self, z_e, z_q, f_mask, i, args):

        recon_x, recon_lattice3 = self.decode(z_e, z_q, f_mask)
        pdb.set_trace()
        recon_lattice1 = self.decode_lattice(recon_x, f_mask).sum(dim=-2)
        # pdb.set_trace()
        recon_lattice2 = self.decode_lattice2(z_q, f_mask).sum(dim=-2)
        recon_lattice = (recon_lattice1 + recon_lattice2 + recon_lattice3) / 3
        recon_x[..., : self.d1] = Func.softmax(recon_x[..., : self.d1], dim=-1)
        lattice_predict = self.normalizer.denorm(recon_lattice)
        structures = self.get_structure_from_recon(
            recon_x, f_mask, lattice_predict, i, args
        )
        return structures

    def get_structure_using_indices(self, indices):
        local_indices, global_indices, lattice_indices = indices
        with torch.no_grad():
            global_z = (
                self.vq2.get_codes_from_indices(global_indices).permute(1, 0, 2).sum(1)
            )
            local_z = (
                self.vq1.get_codes_from_indices(local_indices)
                .permute(1, 0, 2, 3)
                .sum(1)
            )
            lattice_z = (
                self.vql.get_codes_from_indices(lattice_indices)
                .permute(1, 0, 2, 3)
                .sum(1)
            )
            z_e = (local_z, global_z, lattice_z)
            z_q = z_e
            f_mask = torch.ones(local_z.shape[:-1], dtype=torch.bool).to(
                global_z.device
            )
            recon_x, recon_lattice3 = self.decode(z_e, z_q, f_mask)
            recon_lattice1 = self.decode_lattice(recon_x, f_mask).sum(dim=-2)
            recon_lattice2 = self.decode_lattice2(z_q, f_mask).sum(dim=-2)
            recon_lattice = (recon_lattice1 + recon_lattice2 + recon_lattice3) / 3
            recon_x[..., : self.d1] = Func.softmax(recon_x[..., : self.d1], dim=-1)
            lattice_predict = self.normalizer.denorm(recon_lattice)
            structures = self.get_structure_from_recon(recon_x, f_mask, lattice_predict)
            return structures

    def get_structure_using_local_indices(self, indices):
        local_indices, global_indices = indices
        with torch.no_grad():
            global_z = (
                self.vq2.get_codes_from_indices(global_indices).permute(1, 0, 2).sum(1)
            )
            local_z = (
                self.vq1.get_codes_from_indices(local_indices)
                .permute(1, 0, 2, 3)
                .sum(1)
            )
            z_e = (local_z, global_z)
            z_q = z_e
            zq2 = z_q[1]
            f_mask = torch.ones(local_z.shape[:-1], dtype=torch.bool).to(
                global_z.device
            )
            recon_x = self.decode_local(z_e, z_q, f_mask)
            recon_x[..., : self.d1] = Func.softmax(recon_x[..., : self.d1], dim=-1)
            recon_from_global = self.decode_from_global(
                z_e[1], zq2 + torch.sum(z_q[0], dim=1, keepdim=False), z_q[0], f_mask
            )
            recon_x = 0.8*recon_x + 0.2*recon_from_global
            structures = self.get_structure_from_recon(
                recon_x, f_mask, torch.tensor([[10, 10, 10, 90, 90, 90]])
            )
            
            return structures

    def get_indices(
        self,
        F: Tensor,
        A: Tensor,
        f_mask: Tensor,
        lattice: Tensor,
        nbr_fea: Tensor,
        nbr_fea_idx: Tensor,
        crystal_atom_idx,
    ):
        with torch.no_grad():
            z_e = self.encode(
                F, f_mask, nbr_fea, nbr_fea_idx, crystal_atom_idx, A, lattice
            )
            indices, z_q, commitment_loss = self.quantize(z_e, f_mask)
            indices1, indices2 = indices
            B = indices1.shape[0]
            result_list = []
            for i in range(B):
                atom_indice = indices1[i][f_mask[i]].cpu().numpy()
                global_indice = indices2[i].cpu().numpy()
                #lattice_indice = indicesl[i].cpu().numpy()
                # pdb.set_trace()
                atom_indice = custom_sort(atom_indice)
                crystal_indice = list(
                    itertools.chain(
                        global_indice.reshape(-1).tolist(),
                        atom_indice.reshape(-1).tolist(),
                    )
                )
                result_list.append(crystal_indice)
            return result_list


def custom_sort(tensor):
    indices = np.arange(len(tensor))
    tuples = [(tuple(vector), idx) for idx, vector in zip(indices, tensor)]
    tuples.sort()
    sorted_indices = [idx for _, idx in tuples]
    sorted_tensor = tensor[sorted_indices]
    return sorted_tensor


def crystal_position_constrain(tensor):
    transformed_tensor = tensor.clone()
    shape = transformed_tensor.shape
    tensor_flat = transformed_tensor.reshape(-1)
    for idx in range(tensor_flat.shape[0]):
        value = tensor_flat[idx]
        if value > 0.9:
            tensor_flat[idx] = 1
        elif value < 0.1:
            tensor_flat[idx] = 0
        elif 0.45 <= value <= 0.55:
            tensor_flat[idx] = 0.5
    transformed_tensor = tensor_flat.reshape(shape)
    return transformed_tensor
