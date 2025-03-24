"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
from plyfile import PlyData

from pointcept.utils.logger import get_root_logger
from pointcept.utils.cache import shared_dict
from .builder import DATASETS
from .defaults import DefaultDataset
from .transform import Compose, TRANSFORMS
from .preprocessing.scannet.meta_data.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)


@DATASETS.register_module()
class PDEDataset(DefaultDataset):
    VALID_ASSETS = [
        "xyz", 
        "opacities", 
        "features_dc", 
        "scales", 
        "rots", 
        "m", 
        "sigma", 
        "w1"
    ]
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
        self,
        **kwargs,
    ):

        super().__init__(**kwargs)

    def get_data_list(self):
        data_list = [
            os.path.join(self.data_root, str(num)+".ply") for num in range(10000)
        ]
        return data_list

    def get_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        name = self.get_data_name(idx)
        if self.cache:
            cache_name = f"pointcept-{name}"
            return shared_dict(cache_name)

        data_dict = {}
        plydata = PlyData.read(data_path)
        data_dict['coord'] = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        data_dict['opacities'] = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        data_dict['features_dc'] = np.zeros((data_dict['coord'].shape[0], 3))
        data_dict['features_dc'][:, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        data_dict['features_dc'][:, 1] = np.asarray(plydata.elements[0]["f_dc_1"])
        data_dict['features_dc'][:, 2] = np.asarray(plydata.elements[0]["f_dc_2"])

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        data_dict['scales'] = np.zeros((data_dict['coord'].shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            data_dict['scales'][:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        data_dict['rots'] = np.zeros((data_dict['coord'].shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            data_dict['rots'][:, idx] = np.asarray(plydata.elements[0][attr_name])
        
        m_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("m")]
        m_names = sorted(m_names, key = lambda x: int(x.split('_')[-1]))
        data_dict['m'] = np.zeros((data_dict['coord'].shape[0], len(m_names)))
        for idx, attr_name in enumerate(m_names):
            data_dict['m'][:, idx] = np.asarray(plydata.elements[0][attr_name])
        w1_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("w1_")]
        w1_names = sorted(w1_names, key = lambda x: int(x.split('_')[-1]))
        data_dict['w1'] = np.zeros((data_dict['coord'].shape[0], len(w1_names)))
        for idx, attr_name in enumerate(w1_names):
            data_dict['w1'][:, idx] = np.asarray(plydata.elements[0][attr_name])
    
        sigma_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("sigma")]
        sigma_names = sorted(sigma_names, key = lambda x: int(x.split('_')[-1]))
        data_dict['sigma'] = np.zeros((data_dict['coord'].shape[0], len(sigma_names)))
        for idx, attr_name in enumerate(sigma_names):
            data_dict['sigma'][:, idx] = np.asarray(plydata.elements[0][attr_name])

        time_func_names = [p.name for p in plydata.elements[1].properties if p.name.startswith("time")]
        time_func_names = sorted(time_func_names, key = lambda x: int(x.split('_')[-1]))
        time_func = []
        for idx, attr_name in enumerate(time_func_names):
            time_func.append(plydata.elements[1][attr_name])
        time_func = np.array(time_func).flatten().reshape([-1,1])
        data_dict['grid_size'] = 1
        data_dict['segment'] = data_dict['coord']
        # whole_data = torch.tensor(np.concatenate([xyz, opacities, features_dc, scales, rots, m, sigma, w1], axis=-1), dtype=torch.float)
        
        return data_dict
