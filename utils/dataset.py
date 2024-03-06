# This file is use to read Hierachal-MovieNet dataset
import numpy as np
import torch
import cv2
import json
from pathlib import Path
from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Tuple, cast
import os
import random
from tqdm import tqdm


class DataAugmentationForVRM(object):
    # VRM : Video Reorder MovieNet
    def __init__(self, args):
        pass

    def __call__(self, image, text):
        return image, text

    def __repr__(self):
        repr = "(DataAugmentationForVRM)"
        return repr

class VideoReorderMovieNetDataFolder(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str, 
            split: str,
            layer: str = '',
            transform: Optional[Callable] = None
        ) -> None:
        
        super().__init__()
        # init
        if split == 'test': self.split = 'test_in_domain'
        if split not in ['train', 'val', 'test_in_domain', 'test_out_domain', 'all', 'human_behavior/in_domain', "human_behavior/out_domain"]: assert False, 'No such split name'
        self.split = split
        
        self.root = Path(root)

        if layer == 'clip' : layer = ''
        if layer not in ['', 'shot', 'scene'] : assert False, 'No such clip name'
        self.layer = layer

        # read clip_id.json
        with open(Path(self.root, 'clip_id.json'), 'r') as f:
            clip_id_json = json.load(f)
        
        self.clip_list = clip_id_json[self.split]

        # read data .pt file
        if self.layer == '':
            self.data = torch.load(Path(self.root, f'{split}.pt'))
        else:
            # self.data = torch.load(Path(self.root, f'{split}_{self.layer}.pt'))
            self.data = torch.load(Path(self.root, f'{split}_clip_shot.pt'))

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.layer == "":
            clip_id = self.clip_list[index]
            return self.data[clip_id]['feature'], self.data[clip_id]['img_id'], self.data[clip_id]['shot_id'], self.data[clip_id]['scene_id']
        
        if self.layer == "shot":
            return self.data[index]['feature'], self.data[index]['img_id']
            return self.data[index]['feature'], self.data[index]['gt_id']

class NewVideoReorderMovieNetDataFolder(torch.utils.data.Dataset):
    def __init__(
            self,
            root: str, 
            split: str,
            layer: str = '',
            transform: Optional[Callable] = None
        ) -> None:
        
        super().__init__()
        # init
        if split == 'test': self.split = 'test_in_domain'
        if split not in ['train', 'val', 'test_in_domain', 'test_out_domain', 'all', 'human_behavior/in_domain', "human_behavior/out_domain"]: assert False, 'No such split name'
        self.split = split
        
        self.root = Path(root)

        if layer == 'clip' : layer = ''
        if layer not in ['', 'shot', 'scene'] : assert False, 'No such clip name'
        self.layer = layer

        # read clip_id.json
        with open(Path(self.root, 'clip_id.json'), 'r') as f:
            clip_id_json = json.load(f)
        
        self.clip_list = clip_id_json[self.split]

        # read data .pt file
        if self.layer == '':
            self.data = torch.load(Path(self.root, f'{split}_new.pt'))
        else:
            self.data = torch.load(Path(self.root, f'{split}_ori_shot.pt'))

        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if self.layer == "":
            clip_id = self.clip_list[index]
            # img_lst = []
            # for img in self.data[clip_id]['img']:
            #     img_lst.append(torch.load(self.data[clip_id]['img']))
            
            # img = torch.load(self.data[clip_id]['img'])
            # return self.data[clip_id]['text'], img, self.data[clip_id]['img_id'], self.data[clip_id]['shot_id'], self.data[clip_id]['scene_id']
            
            return self.data[clip_id]['text'], self.data[clip_id]['img'], self.data[clip_id]['img_id'], self.data[clip_id]['shot_id'], self.data[clip_id]['scene_id']
        
        if self.layer == "shot":
            
            return self.data[index]['text'], self.data[index]['img'], self.data[index]['label']



class VideoReorderMovieNetDataLoader(object):
    def __init__(self) -> None:
        pass

    def __next__(self):
        return


def build_VideoReorderMovieNet_dataset(args):
    '''
    return  list[list[], ...]
    the fisrt list is batch
    the second is [['feature'], ['shot_id'], ['scene_id]]
    '''
    transform = DataAugmentationForVRM(args)
    print("Data Aug = %s" % str(transform))
    root = args.data_path
    split = args.split
    return VideoReorderMovieNetDataFolder(root, split, transform=transform, collate_fn=lambda x: x)

if __name__ == '__main__':
    pass