# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
from collections import defaultdict
import pickle
from tqdm import tqdm
import random
import os
import shutil

class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'Market-1501-v15.09.15'

    def __init__(self, root='', verbose=True, pid_begin = 0, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        self.pid_begin = pid_begin
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> Market1501 loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        #breakpoint()

        # Trim experiments
        #self.query = self.remove_camid(self.query, [11, 12, 13], keep_only=True)
        #self.gallery = self.remove_camid(self.gallery, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 14], keep_only=True)
        
        self.train = self.remove_camid(self.train, [1, 2, 3, 4, 5], keep_only=True, unique_pids=True)
        self.query = self.remove_camid(self.query, [0], keep_only=True)
        self.gallery = self.remove_camid(self.gallery, [1, 2, 3, 4, 5], keep_only=True)
        
        #self.train = self.reduce_train_data(self.train)
        #self.query = self.remove_camid(self.train, [0], keep_only=True)
        #self.train = self.remove_camid(self.train, [11, 12, 13], keep_only=True)
        
        if verbose:
            print("=> MARKET1501 statistics after trimming")
            self.print_dataset_statistics(self.train, self.query, self.gallery)

        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_vids = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_vids = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_vids = self.get_imagedata_info(self.gallery)

    def remove_camid(self, dataset, camids, keep_only=False, unique_pids=False):
        camids = list(set(camids))
        new_dataset = []
        if unique_pids:
            pids = set()
        else: pids = []
        for d in tqdm(dataset, total=len(dataset), desc="Trimming dataset"):
            
            #breakpoint()
            cid = d[2]
            if keep_only:
                if cid in camids: 
                    if unique_pids: pids.add(d[1])
                    else: pids.append(d[1])
                    new_dataset.append(d)
            else:
                if cid not in camids: 
                    if unique_pids: pids.add(d[1])
                    else: pids.append(d[1])
                    new_dataset.append(d)

        if unique_pids:
            pids = list(pids)
            for idx in range(len(new_dataset)):
                new_dataset[idx] = (new_dataset[idx][0], pids.index(new_dataset[idx][1]), new_dataset[idx][2], new_dataset[idx][3])
        else: 
            for idx in range(len(new_dataset)):
                new_dataset[idx] = (new_dataset[idx][0], new_dataset[idx][1], new_dataset[idx][2], new_dataset[idx][3])

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
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in sorted(img_paths):
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}
        dataset = []
        for img_path in sorted(img_paths):
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            dataset.append((img_path, self.pid_begin + pid, camid, 0))
        return dataset