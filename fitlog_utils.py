
import fitlog 
import torch.distributed as dist


def add_hyper(params):
    if dist.get_rank() == 0:
        tmp_dict={}
        for k,v in params.items():
            if isinstance(v,tuple) or isinstance(v,list):
                tmp_dict[k]=v
        fitlog.add_hyper(params)
        for k,v in tmp_dict.items():
            params[k]=v

def init_fitlog(param_grid, log_dir='logs'):
    if dist.get_rank() == 0:
        fitlog.commit(__file__)
        fitlog.set_log_dir(log_dir)
        add_hyper(param_grid)

import os

def create_folder(folder_path):
    try:
        # Attempt to create the directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        print(f"create directory: {folder_path}")
    except Exception as e:
        print(f"Failed to create directory {folder_path}: {str(e)}")

