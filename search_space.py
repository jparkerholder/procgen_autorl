
import pandas as pd
from auto_drac import data_augs

def get_hparams(args):
    """
    Only supports the data aug experiment
    Should be trivial to add more...
    """
    if args.experiment == "data_aug":
        
        if args.fixed:
            names = ['aug_type']
            types = ['categorical']
            ranges = [['crop']]
        else:
            names = ['entropy_coef', 'lr', 'clip_param', 'aug_coef', 'aug_type']
            types = ['continuous', 'continuous', 'continuous', 'continuous', 'categorical']
            ranges = [[0,0.2], 
                      [1e-5, 1e-3], 
                      [0.01, 0.5],
                      [0.01, 0.5],
                      ['crop',
                       'random-conv',
                       'grayscale',
                       'flip',
                       'rotate',
                       'cutout',
                       'cutout-color',
                       'color-jitter']]
        
    df_hparams = pd.DataFrame({
        'Name': names,
        'Type': types,
        'Range': ranges})
    
    return(df_hparams)


aug_to_func = {    
        'crop': data_augs.Crop,
        'random-conv': data_augs.RandomConv,
        'grayscale': data_augs.Grayscale,
        'flip': data_augs.Flip,
        'rotate': data_augs.Rotate,
        'cutout': data_augs.Cutout,
        'cutout-color': data_augs.CutoutColor,
        'color-jitter': data_augs.ColorJitter,
}
