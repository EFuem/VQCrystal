import torch
from scipy.optimize import linear_sum_assignment
import numpy as np
import pdb

@torch.no_grad()
def similarity_matrix(A, A_, F, F_, M, d1, d2):
    B, N, _ = A.shape
    _, K, _ = A_.shape
    # most cases
    A_diagnol = A_.diagonal(dim1=1, dim2=2) # (B, K)
    A_diagnol_cross = A_diagnol.unsqueeze(2) @ A_diagnol.unsqueeze(1)
    A_diagnol_cross = A_ * A_diagnol_cross
    S = torch.einsum('bac,bkp->bakcp', A, A_diagnol_cross).view(A.size(0), 
                                                     A.size(1)*A_diagnol_cross.size(1), 
                                                     A.size(2)*A_diagnol_cross.size(2) 
                                                     ) # (B, NK, NK)
    # i=j, a=b
    V = F[:, :,:d1]
    P = F[:, :,d1:]
    V_ = F_[:, :,:d1]
    P_ = F_[:, :,d1:]
    correlation1 = V @ torch.transpose(V_, 1,2).contiguous()
    correlation1 = correlation1.view(B, -1)
    P_ = P_.repeat(1, N, 1)
    P = P.unsqueeze(dim = 1) # B, 1, N, D
    P = P.repeat(1, K, 1, 1)
    P = torch.transpose(P, 1, 2).contiguous() # B, N, K, D
    P = P.view(B, N*K, d2)
    correlation2 = torch.norm(P - P_, dim=-1)
    correlation2 = correlation2.view(B, -1)
    correlation2 = 1 / (1+correlation2)
    correlation = correlation1 * correlation2
    # existence
    A_diagnol = A_diagnol.repeat(1, N)
    correlation = correlation * A_diagnol
    M = M.unsqueeze(dim=1) # B, 1, N
    M = M.repeat(1, K, 1)
    M = torch.transpose(M, 1, 2).contiguous()
    M = M.view(B, N*K)
    correlation = correlation * M
    S_diagonal = torch.diag_embed(correlation)
    mask = torch.eye(N*K).bool().to(S.device)
    S = torch.where(mask, S_diagonal, S)
    return S

@torch.no_grad()
def max_pooling_matching(S, N, K, M, num_iter=5):
    # initialize x
    B = S.shape[0]
    x = torch.ones((B, N*K)).to(S.device)
    M_ = M.unsqueeze(dim=1) # B, 1, N
    M_ = M_.repeat(1, K, 1)
    M_ = torch.transpose(M_, 1, 2).contiguous()
    M_ = M_.view(B, N*K)
    x = x * M_
    x_norm = torch.norm(x, dim=1, keepdim=True)
    x = x / x_norm

    S_diagnol = S.diagonal(dim1=1, dim2=2) # (B, NK)
    S_diagnol_sub = torch.zeros_like(S_diagnol, device=S.device)
    S_diagnol_sub = torch.diag_embed(S_diagnol_sub)
    mask = torch.eye(N*K).bool().to(S.device)
    S_res = torch.where(mask, S_diagnol_sub, S)
    for i in range(num_iter):
        res1 = x * S_diagnol
        high_x = x.unsqueeze(-1).repeat(1,1,N*K)
        res2 = S_res * high_x # B, NK, NK
        res_2 = res2.view(B, N*K, N, K)
        res_2 = torch.sum(torch.max(res_2, dim=-1)[0], dim=-1) # B, NK
        res = res1 + res_2
        res_norm = torch.norm(res, dim=1, keepdim=True)
        x = res / res_norm
    # Hungarian Algorithm
    X_ = x.view(B, N, K).cpu().numpy()
    M_ = M.sum(dim=-1).cpu().tolist()
    X_list = []
    for i in range(B):
        X_s = X_[i, :, :]
        row_ind, col_ind = linear_sum_assignment(X_s, maximize=True)
        X_s_array = np.zeros_like(X_s)
        for j in range(M_[i]):
            X_s_array[row_ind[j], col_ind[j]] = 1
        X_list.append(X_s_array)
    X_array = np.array(X_list)
    X = torch.from_numpy(X_array).to(S.device)
    return torch.transpose(X, 1, 2).contiguous() # B, K, N