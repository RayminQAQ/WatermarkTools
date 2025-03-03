"""
File: get_models.py
Usage: model = get_models(args, sample)
Ref: 
    - VAE collection: https://github.com/AntixK/PyTorch-VAE
    - Diffusion model collection: https://github.com/openai/guided-diffusion
    - ?
"""
import torch
from .simpleVAE import SimpleVAE 


def get_models(args, sample: torch.tensor) -> torch.nn.Module:
    """
    param:
    - sample: for initializing models
    """

    # Note: the parameter may varies
    MODEL_MAP = {
        "SimpleVAE": SimpleVAE(sample=sample, embed_dim=512, latent_dim=128), 
        
        # TBD ...
        
        # ADD MODEL HERE
    }
    
    model = MODEL_MAP[args.model_name_or_path] # Allow one model at a time 
    
    # Warning
    if args.model_name_or_path not in MODEL_MAP:
        raise ValueError(f"WARNING: Dataset name _{args.model}_ is not support by function get_models()")
        
    return model