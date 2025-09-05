import torch
import numpy as np

def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_counterfactual(base_prompt, cf_prompt, codegen_func, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    orig_out = codegen_func(base_prompt)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cf_out = codegen_func(cf_prompt)
    return orig_out, cf_out 