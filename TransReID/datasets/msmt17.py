
import glob
import re
from tqdm import tqdm
import random

import os.path as osp

from .bases import BaseImageDataset


class MSMT17(BaseImageDataset):
    """
    MSMT17

    Reference:
    Wei et al. Person Transfer GAN to Bridge Domain Gap for Person Re-Identification. CVPR 2018.

    URL: http://www.pkuvmc.com/publications/msmt17.html

    Dataset statistics:
    # identities: 4101
    # images: 32621 (train) + 11659 (query) + 82161 (gallery)
    # cameras: 15
    """
    dataset_dir = 'MSMT17'

    def __init__(self, root='', verbose=True, pid_begin=0, **kwargs):
        super(MSMT17, self).__init__()
        self.pid_begin = pid_begin
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.test_dir = osp.join(self.dataset_dir, 'test')
        self.list_train_path = osp.join(self.dataset_dir, 'list_train.txt')
        self.list_val_path = osp.join(self.dataset_dir, 'list_val.txt')
        self.list_query_path = osp.join(self.dataset_dir, 'list_query.txt')
        self.list_gallery_path = osp.join(self.dataset_dir, 'list_gallery.txt')

        self._check_before_run()
        train = self._process_dir(self.train_dir, self.list_train_path)
        val = self._process_dir(self.train_dir, self.list_val_path)
        train += val
        query = self._process_dir(self.test_dir, self.list_query_path)
        gallery = self._process_dir(self.test_dir, self.list_gallery_path)
        if verbose:
            print("=> MSMT17 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        #breakpoint()
        # Trim experiments
        #self.query = self.remove_camid(self.query, [11, 12, 13], keep_only=True)
        #self.train = self.remove_camid(self.train, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], keep_only=True)
        self.train = self.remove_camid(self.train, [1, 2, 3, 4, 5,], keep_only=True)
        #self.train = self.reduce_train_data(self.train)
        #self.query = self.remove_camid(self.train, [0], keep_only=True)
        #self.train = self.remove_camid(self.train, [11, 12, 13], keep_only=True)
        if verbose:
            print("=> MSMT17 statistics after trimming")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)
    
    def remove_camid(self, dataset, camids, keep_only=False):
        new_dataset = []
        pids = set()
        for d in tqdm(dataset, total=len(dataset), desc="Trimming dataset"):
            
            #breakpoint()
            cid = d[2]
            if keep_only:
                if cid in camids: 
                    pids.add(d[1])
                    new_dataset.append(d)
            else:
                if cid not in camids: 
                    pids.add(d[1])
                    new_dataset.append(d)

        pids = list(pids)
        for idx in range(len(new_dataset)):
            new_dataset[idx] = (new_dataset[idx][0], pids.index(new_dataset[idx][1]), new_dataset[idx][2], new_dataset[idx][3])

        return new_dataset
    
    def reduce_train_data(self, dataset, num_per_cam=560):
        dataset_dict, new_dataset = {}, []
        new_pids = set()
        for d in tqdm(dataset, total=len(dataset), desc="Trimming dataset"):
            cid = d[2]
            if cid not in list(dataset_dict.keys()):
                dataset_dict[cid] = [d]
            else: dataset_dict[cid].append(d)
        
        for k,v in dataset_dict.items():
            print(f"Cam ID: {k}, Total: {len(v)}")
            random.shuffle(dataset_dict[k])
            dataset_dict[k] = dataset_dict[k][:num_per_cam]
            for d in dataset_dict[k]:
                new_pids.add(d[1])

        new_pids = list(new_pids)
        for k,v in dataset_dict.items():
            print(f"Cam ID: {k}, Total: {len(v)}")
            for d in dataset_dict[k]:
                new_dataset.append((d[0], new_pids.index(d[1]), d[2], d[3]))
        
        random.shuffle(new_dataset)
        return new_dataset

    def show_per_cam_stats(self, dataset):
        samples_per_cam = {}
        pids_per_cam = {}
        for i in range(15): 
            samples_per_cam[i] = []
            pids_per_cam[i] = set()
        
        for d in dataset:
            samples_per_cam[d[2]].append(d)
            pids_per_cam[d[2]].add(d[1])
        
        for i in range(15):
            print(f"CamID: {i}, Number of samples: {len(samples_per_cam[i])}, Number of PIDs: {len(list(pids_per_cam[i]))}")

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.test_dir):
            raise RuntimeError("'{}' is not available".format(self.test_dir))

    def _process_dir(self, dir_path, list_path):
        with open(list_path, 'r') as txt:
            lines = txt.readlines()
        dataset = []
        pid_container = set()
        cam_container = set()
        for img_idx, img_info in enumerate(lines):
            img_path, pid = img_info.split(' ')
            pid = int(pid)  # no need to relabel
            camid = int(img_path.split('_')[2])
            img_path = osp.join(dir_path, img_path)
            dataset.append((img_path,  self.pid_begin +pid, camid-1, 1))
            pid_container.add(pid)
            cam_container.add(camid)
        print(cam_container, 'cam_container')
        # check if pid starts from 0 and increments with 1
        for idx, pid in enumerate(pid_container):
            assert idx == pid, "See code comment for explanation"
        return dataset