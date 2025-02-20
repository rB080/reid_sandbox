import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import os.path as osp 
import json
from tqdm import tqdm
import random

from datasets.bases import read_image
from utils.metrics import *
from analyse import *


from collections import Counter



# Test time adapter:

class TestTimeAdapter(nn.Module):
    def __init__(self, dims, gallery_feats, gallery_camids, steps=1, device='cuda:1', lr=0.0005, topk=50):
        super(TestTimeAdapter, self).__init__()
        self.gallery_feats = gallery_feats
        self.gallery_camids = gallery_camids
        self.gnorms = gnorms
        self.qnorms = qnorms
        
        self.steps = steps
        self.lr = lr
        self.topk = topk
        self.reset()
        self.device = device
        
        
    def loss(self, q, g):
        S = euclidean_distance(q, g, return_tensor=True)
        loss = 0.0
        for i in range(S.shape[0]):
            s = S[i]
            # Find the indices of the k smallest values
            _, topk_indices = torch.topk(s, self.topk, largest=False)
            # Create a mask with all False
            mask = torch.zeros_like(s, dtype=torch.bool)
            # Set True for top-k smallest indices
            mask[topk_indices] = True
            # Set all non-top-k elements to zero (or another value)
            s[~mask] = 0  # In-place operation
            loss += s.sum()
        return loss / S.shape[0]
    
    def reset(self):
        self.param_list = []
        for k,v in self.gnorms.items():
            self.gnorms[k] = [torch.nn.Parameter(v[0]), torch.nn.Parameter(v[1])]
            #self.gnorms[k] = [v[0], v[1]]
            #self.qnorms[k] = [torch.nn.Parameter(qnorms[k][0]), torch.nn.Parameter(qnorms[k][1])]
            self.qnorms[k] = [qnorms[k][0], qnorms[k][1]]
            self.param_list.append(self.gnorms[k][0])
            self.param_list.append(self.gnorms[k][1])
            # self.param_list.append(self.qnorms[k][0])
            # self.param_list.append(self.qnorms[k][1])
        self.optimizer = torch.optim.Adam(params=self.param_list, lr=self.lr, weight_decay=1e-4)

    def forward(self, x, c, episodic=True):
        if episodic: self.reset()
        
        for _ in range(self.steps):
            x_norm, gf_norm = forward_and_adapt(self, x, c)
        

        return x_norm, gf_norm
    
@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(adapter, x, c):
    #breakpoint()
    g = adapter.gallery_feats
    # Gallery normalization
    means, stds = [], []
    for i in adapter.gallery_camids:
        means.append(adapter.gnorms[i][0])
        stds.append(adapter.gnorms[i][1])
    means = torch.stack(means)
    stds = torch.stack(stds)
    gf_norm = (g - means.to(adapter.device)) / stds.to(adapter.device)

    # Query normalization
    means, stds = [], []
    for i in c:
        means.append(adapter.qnorms[i][0])
        stds.append(adapter.qnorms[i][1])
    means = torch.stack(means)
    stds = torch.stack(stds)
    #print(means.shape, stds.shape)
    x_norm = (x - means.to(adapter.device)) / stds.to(adapter.device)
    #print(x.shape, x_norm.shape, gf_norm.shape)
    loss = adapter.loss(x_norm, gf_norm)
    loss.backward()
    adapter.optimizer.step()
    adapter.optimizer.zero_grad()

    return x_norm, gf_norm

# Tester
def test_grid(steps, lr, topk, pids, camids, qf, gf, batchsize=32):

    adapter = TestTimeAdapter(dims=1280, gallery_feats=gf.to('cuda:1'), gallery_camids=camids[qf.shape[0]:], steps=steps, lr=lr, topk=topk).to('cuda:1').train()

    qf_new = []
    batchsize = 32
    total_batches = qf.shape[0] // batchsize
    for i in tqdm(range(total_batches + 1)):
        if batchsize * i == qf.shape[0]: break
        q = qf[batchsize * i:(i+1)*batchsize].to('cuda:1')
        c = camids[:qf.shape[0]][batchsize * i:(i+1)*batchsize]
        q, g = adapter(q, c)
        #print(q.shape, g.shape)
        qf_new.append(q.detach().cpu())
        gf_new = g.detach().cpu()
    qf_new = torch.cat(qf_new, dim=0)
    #print(qf_new.shape)
    # fetch from bins

    query_feats, gallery_feats = qf_new, gf_new
    PIDS, CIDS = pids, camids
    #print(qf.shape, gf.shape, indices.shape)
    evaluator = R1_mAP_eval(num_query=query_feats.shape[0], max_rank=50, feat_norm=True)
    feats = torch.cat([query_feats, gallery_feats], dim=0)
    evaluator.feats = feats
    evaluator.pids = PIDS
    evaluator.camids = CIDS
    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    print(f"mAP: {mAP}, R1: {cmc[0]}, R5: {cmc[4]}, R10: {cmc[9]}, R50: {cmc[49]}")
    torch.cuda.empty_cache()
    del adapter




if __name__ == "__main__":

    # Load data and find normalization parameters
    print("Loading data!")
    output_dir = "/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/check_features"
    feature_type = "tent2"
    pids = torch.load(osp.join(output_dir, "pids.pth"))
    camids = torch.load(osp.join(output_dir, "camids.pth"))
    with open(osp.join(output_dir, "imgpaths.json"), 'r') as f:
        file_content = f.read()  # Read the entire content of the file as a string
        imgpaths = json.loads(file_content) 
    Q, G = torch.load(osp.join(output_dir, f"qf_{feature_type}.pth")), torch.load(osp.join(output_dir, f"gf_{feature_type}.pth"))
    print("Loaded successfully!")


    qbins, gbins = {}, {}
    qpids, gpids = {}, {}
    for i in range(Q.shape[0]):
        if camids[i] in qbins: 
            qbins[camids[i]].append(Q[i].unsqueeze(0))
            qpids[camids[i]].append(pids[i])
        else: 
            qbins[camids[i]] = [Q[i].unsqueeze(0)]
            qpids[camids[i]] = [pids[i]]

    for i in range(G.shape[0]):
        if camids[Q.shape[0] + i] in gbins: 
            gbins[camids[Q.shape[0] + i]].append(G[i].unsqueeze(0))
            gpids[camids[Q.shape[0] + i]].append(pids[Q.shape[0] + i])
        else: 
            gbins[camids[Q.shape[0] + i]] = [G[i].unsqueeze(0)]
            gpids[camids[Q.shape[0] + i]] = [pids[Q.shape[0] + i]]

    qnorms, gnorms = {}, {}
    for k,v in qbins.items():
        qnorms[k] = [torch.cat(v, dim=0).mean(dim=0), torch.cat(v, dim=0).std(dim=0)]
    for k,v in gbins.items():
        gnorms[k] = [torch.cat(v, dim=0).mean(dim=0), torch.cat(v, dim=0).std(dim=0)]
    ############################################################################################################################################


    S = [1, 2, 5]
    L = [0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    T = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200]

    for s in S:
        for l in L:
            for t in T:
                print("================================================================================")
                print(f"Steps: {s}, LR: {l}, Topk: {t}")
                test_grid(steps=s, lr=l, topk=t, pids=pids, camids=camids, qf=Q, gf=G)
                print("================================================================================")