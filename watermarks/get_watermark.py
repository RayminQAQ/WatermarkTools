"""
TODO:
    - Watermark algo-1: DwtDctSvd, RivaGAN, SSL, StableSignature, StegaStamp, TreeRing (same with https://arxiv.org/html/2410.05470v1)
    - Watermark algo-2: add noise for our own
    
Resourse:
    - https://github.com/ShieldMnt/invisible-watermark/tree/main
    - https://github.com/XuandongZhao/WatermarkAttacker/blob/main/wmattacker.py#L68
    - https://blog.csdn.net/weixin_42662358/article/details/90448566
    
"""
import torch

def get_watermarks(args, images: torch.tensor | list[torch.tensor]):
    
    WATERMARK_MAP = {
        #"watermarkName": algo(dataset),
    }
    
    new_dataset = WATERMARK_MAP[args.watermark] if args.watermark in WATERMARK_MAP else None
    if new_dataset is None:
        raise ValueError("WARNING: get_watermarks() is not use")
        
    return new_dataset

if __name__ == "__main__":
    # Testing Area
    pass