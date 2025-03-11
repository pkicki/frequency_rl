import numpy as np
import torch

def project_entropy_independently(chol, e_lb):
    a_dim = chol.size()[-1]
    c = a_dim / 2 * np.log(2 * np.pi * np.e)
    avg_log_diag = (e_lb - c) / a_dim
    chol_diag = torch.maximum(chol.diagonal(dim1=-2, dim2=-1).log(), torch.tensor(avg_log_diag)).exp()
    chol_ = torch.diag_embed(chol_diag, dim1=-2, dim2=-1)
    return chol_


def project_entropy(chol, e_lb):
    a_dim = chol.size()[-1]
    def entropy(chol):
        return a_dim / 2 * np.log(2 * np.pi * np.e) + torch.diagonal(chol, dim1=-2, dim2=-1).log().sum(-1)
    ent = entropy(chol)
    chol = torch.where(ent < e_lb, chol * torch.exp((e_lb - ent) / a_dim), chol)
    return chol