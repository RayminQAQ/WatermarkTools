"""
TODO:
    - Watermark algo-1: DwtDctSvd, RivaGAN, SSL, StableSignature, StegaStamp, TreeRing
    - Watermark algo-2: add noise for our own
"""

def get_watermarks(args, dataset):
    
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