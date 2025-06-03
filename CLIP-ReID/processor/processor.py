import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist

from tqdm import tqdm
import json 
from torchvision.utils import save_image
from tent.tdist2 import *
from tent.tdist3 import *


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            # breakpoint()
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view)
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            if cfg.MODEL.SIE_CAMERA:
                                camids = camids.to(device)
                            else: 
                                camids = None
                            if cfg.MODEL.SIE_VIEW:
                                target_view = target_view.to(device)
                            else: 
                                target_view = None
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        if cfg.MODEL.SIE_CAMERA:
                            camids = camids.to(device)
                        else: 
                            camids = None
                        if cfg.MODEL.SIE_VIEW:
                            target_view = target_view.to(device)
                        else: 
                            target_view = None
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Total running time: {}".format(total_time))
    print(cfg.OUTPUT_DIR)

def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []

    for n_iter, (img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)


    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


def do_train_modified(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            with amp.autocast(enabled=True):
                score, feat = model(img, target, cam_label=target_cam, view_label=target_view )
                loss = loss_fn(score, feat, target, target_cam)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    model.eval()
                    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        with torch.no_grad():
                            img = img.to(device)
                            camids = camids.to(device)
                            target_view = target_view.to(device)
                            feat = model(img, cam_label=camids, view_label=target_view)
                            evaluator.update((feat, vid, camid))
                    cmc, mAP, _, _, _, _, _ = evaluator.compute()
                    logger.info("Validation Results - Epoch: {}".format(epoch))
                    logger.info("mAP: {:.1%}".format(mAP))
                    for r in [1, 5, 10]:
                        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                    torch.cuda.empty_cache()
            else:
                model.eval()
                for n_iter, (img, segimg, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = img.to(device)
                        camids = camids.to(device)
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
                torch.cuda.empty_cache()


def do_inference_modified(cfg,
                 model,
                 val_loader,
                 num_query, segmentor=None, path=None, save_sample=False, camera_normalize=False):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        if segmentor is not None: segmentor.to(device)

    model.eval()
    img_path_list = []
    cam_dict = {}

    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            if segmentor is not None:
                #breakpoint()
                seg_img.to(device)
                prompt = ["the person in the image"] * seg_img.shape[0]
                mask = segmentor(seg_img, prompt)[0]
                mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                mask = torch.sigmoid(mask)
                #breakpoint()
                threshold = 0.5
                mask[mask >= threshold] = 1.0
                mask[mask < threshold] = 0.0
                img = img * (1 - mask)
            camids = camids.to(device)
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            if camera_normalize:
                for cid in camid:
                    if cid in list(cam_dict.keys()):
                        cam_dict[cid].append(feat.detach().cpu())
                    else:
                        cam_dict[cid] = [feat.detach().cpu()]
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            if save_sample:
                img_new = img.detach().cpu()
                img1 = img_new[0]
                save_image(img1, os.path.join(path, "sample.png"))

    if camera_normalize:
        for key, feat in cam_dict.items():
            cam_dict[key] = torch.cat(feat, 0)
            cam_dict[key] = [cam_dict[key].mean(0), cam_dict[key].std(0)]
        feats = torch.cat(evaluator.feats, dim=0)
        means, stds = [], []
        #breakpoint()
        for i in range(feats.shape[0]):
            means.append(cam_dict[evaluator.camids[i]][0])
            stds.append(cam_dict[evaluator.camids[i]][1])
        means = torch.stack(means)
        stds = torch.stack(stds)
        #breakpoint()
        feats = (feats - means) / stds
        evaluator.feats = feats

    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    #breakpoint()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_og.pth")
        print(os.path.isfile(f"{path}/distmat_og.pth"))
        print(f"{path}/distmat_og.pth")
        torch.save(qf, f"{path}/qf_og.pth")
        torch.save(gf, f"{path}/gf_og.pth")
        with open(f"{path}/imgpaths.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids.pth")
        torch.save(camids, f"{path}/camids.pth")
    
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]



def do_tta_inference(cfg,
                 model,
                 qr_loader,
                 gal_loader,
                 tta_args,
                 num_query, segmentor=None, path=None, suffix='tent2', 
                 save_sample=False, camera_normalize=False, compute_entropy=False, new_query=False, single_query = None, mode="ours"):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=False)#cfg.TEST.FEAT_NORM)
    evaluator.reset()
    query_feats = []
    qpids, qcamids = [], []
    gallery_feats = []
    gpids, gcamids = [], []
    new_query_feats = []
    new_qpids, new_qcamids = [], []

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        if segmentor is not None: segmentor.to(device)

    model.eval()
    
    img_path_list = []
    cam_dict = {}
    entropies = []
    iterator = tqdm(enumerate(gal_loader), total=len(gal_loader), desc="Gallery inferencing")
    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in iterator:
        with torch.no_grad():
            #breakpoint()
            img = img.to(device)
            if segmentor is not None:
                #breakpoint()
                seg_img.to(device)
                prompt = ["the person in the image"] * seg_img.shape[0]
                mask = segmentor(seg_img, prompt)[0]
                mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                mask = torch.sigmoid(mask)
                #breakpoint()
                threshold = 0.5
                mask[mask >= threshold] = 1.0
                mask[mask < threshold] = 0.0
                img = img #* (1.0 - mask)
            
        if cfg.MODEL.SIE_CAMERA:
            camids = camids.to(device)
        else: 
            camids = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else: 
            target_view = None
        feat = model(img, cam_label=camids, view_label=target_view)
        gallery_feats.append(feat.detach().cpu())
        gpids.extend(pid)
        gcamids.extend(camid)
        
        img_path_list.extend(imgpath)
    
    gallery_feats = torch.cat(gallery_feats, dim=0)
    if new_query:
        iterator = tqdm(enumerate(qr_loader), total=len(qr_loader), desc="Query inferencing")
        for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in iterator:
            with torch.no_grad():
                #breakpoint()
                img = img.to(device)
                if segmentor is not None:
                    #breakpoint()
                    seg_img.to(device)
                    prompt = ["the person in the image"] * seg_img.shape[0]
                    mask = segmentor(seg_img, prompt)[0]
                    mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                    mask = torch.sigmoid(mask)
                    #breakpoint()
                    threshold = 0.5
                    mask[mask >= threshold] = 1.0
                    mask[mask < threshold] = 0.0
                    img = img #* (1.0 - mask)
                
            if cfg.MODEL.SIE_CAMERA:
                camids = camids.to(device)
            else: 
                camids = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            feat = model(img, cam_label=camids, view_label=target_view)
            new_query_feats.append(feat.detach().cpu())
            new_qpids.extend(pid)
            new_qcamids.extend(camid)
            
            img_path_list.extend(imgpath)
        
        new_query_feats = torch.cat(new_query_feats, dim=0)
        ref_feats = torch.cat([new_query_feats, gallery_feats], dim=0)
        ref_pids = new_qpids + gpids
        ref_camids = new_qcamids + gcamids
        qsize = new_query_feats.shape[0]


    model = configure_model(model)
    params, param_names = collect_params(model)
    #breakpoint()
    #optimizer = torch.optim.Adam(params=params, lr=0.001, weight_decay=1e-4)
    #tented_model = Tent(model, optimizer, steps=1, episodic=True)
    print("Initializing Test Time Wrapper")
    # tdisted_model = beta_TDIST(model, steps=1, device='cuda', lr=0.0001, topk=10, temp=600.0, episodic=False, lite=False, use_norm=True)
    tdisted_model = beta_TDIST(model, **tta_args, mode=mode, transreid=True)
    if not new_query: tdisted_model.find_norms(gallery_feats, gcamids)
    else: tdisted_model.find_norms(ref_feats, ref_camids, clipsize=qsize)
    tdisted_model.initialize(params)
    tdisted_model.train()

    iterator = tqdm(enumerate(qr_loader), total=len(qr_loader))
    
    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in iterator:
        #with torch.no_grad():
        #breakpoint()
        img = img.to(device)
        if segmentor is not None:
            #breakpoint()
            seg_img.to(device)
            prompt = ["the person in the image"] * seg_img.shape[0]
            mask = segmentor(seg_img, prompt)[0]
            mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
            mask = torch.sigmoid(mask)
            #breakpoint()
            threshold = 0.5
            mask[mask >= threshold] = 1.0
            mask[mask < threshold] = 0.0
            img = img #* (1.0 - mask)
            
        if cfg.MODEL.SIE_CAMERA:
            camids = camids.to(device)
        else: 
            camids = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else: 
            target_view = None
        #breakpoint()
        # feat = model(x=img, text=None, batch_size=img.shape[0])
        # feat = torch.matmul(feat, featt.T)
        # feat = model(text="A photo of a person", batch_size=img.shape[0]) 
        #print(imgpath[0])
        if not compute_entropy: 
            feat, gfeat, loss = tdisted_model(img, cam_label=camid, view_label=target_view)
            iterator.set_description(f"Batch Loss: {loss:.3f}")
        else:
            feat, gfeat = tdisted_model(img, cam_label=camid, view_label=target_view)
            #breakpoint()
            _, logits = tdisted_model.model.TTA_forward(img)
            entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
            entropies.extend(entropy.detach().cpu())
        query_feats.append(feat.cpu())
        gallery_feats = gfeat.cpu()
        qpids.extend(pid)
        qcamids.extend(camid)

        # breakpoint()
        #evaluator.update((feat, pid, camid))
        img_path_list.extend(imgpath)
        if save_sample:
            img_new = img.detach().cpu()
            img1 = img_new[0]
            save_image(img1, os.path.join(path, "sample.png"))
    
    query_feats = torch.cat(query_feats, dim=0)

    if single_query is not None:
        print(f"removing everything except query camID {single_query}")
        qf, qp, qc = [], [], []
        for i in range(query_feats.shape[0]):
            if qcamids[i] in single_query:
                qf.append(query_feats[i])
                qp.append(qpids[i])
                qc.append(qcamids[i])
        query_feats = torch.stack(qf)
        qpids = qp
        qcamids = qc
        print(f"New query size: {query_feats.shape[0]} vs Old query size: {num_query}")
        num_query = query_feats.shape[0]

    feats = torch.cat([query_feats, gallery_feats], dim=0)
    evaluator.feats = feats
    evaluator.pids = qpids + gpids
    evaluator.camids = qcamids + gcamids
    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_{suffix}.pth", pickle_protocol=4)
        torch.save(qf, f"{path}/qf_{suffix}.pth", pickle_protocol=4)
        torch.save(gf, f"{path}/gf_{suffix}.pth", pickle_protocol=4)
        with open(f"{path}/imgpaths_{suffix}.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids_{suffix}.pth", pickle_protocol=4)
        torch.save(camids, f"{path}/camids_{suffix}.pth", pickle_protocol=4)

    if compute_entropy:
        E_mean = torch.stack(entropies).mean()
        E_std = torch.stack(entropies).std()
        logger.info("Entropy mean: {:.3f}, Entropy std: {:.3f}".format(E_mean, E_std))
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]

def do_tta_inference_lite(cfg,
                 model,
                 qr_loader,
                 gal_loader,
                 tta_args,
                 num_query, segmentor=None, path=None, suffix='tent2', 
                 save_sample=False, camera_normalize=False, compute_entropy=False, new_query=False, single_query = None):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=False)#cfg.TEST.FEAT_NORM)
    evaluator.reset()
    query_feats = []
    qpids, qcamids = [], []
    gallery_feats = []
    gpids, gcamids = [], []
    new_query_feats = []
    new_qpids, new_qcamids = [], []
    batch_size = 32

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)
        if segmentor is not None: segmentor.to(device)

    model.eval()
    
    img_path_list = []
    cam_dict = {}
    entropies = []
    iterator = tqdm(enumerate(gal_loader), total=len(gal_loader), desc="Gallery inferencing")
    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in iterator:
        with torch.no_grad():
            #breakpoint()
            img = img.to(device)
            if segmentor is not None:
                #breakpoint()
                seg_img.to(device)
                prompt = ["the person in the image"] * seg_img.shape[0]
                mask = segmentor(seg_img, prompt)[0]
                mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                mask = torch.sigmoid(mask)
                #breakpoint()
                threshold = 0.5
                mask[mask >= threshold] = 1.0
                mask[mask < threshold] = 0.0
                img = img #* (1.0 - mask)
            
        if cfg.MODEL.SIE_CAMERA:
            camids = camids.to(device)
        else: 
            camids = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else: 
            target_view = None
        feat = model(img, cam_label=camids, view_label=target_view)
        gallery_feats.append(feat.detach().cpu())
        gpids.extend(pid)
        gcamids.extend(camid)
        
        img_path_list.extend(imgpath)
    
    gallery_feats = torch.cat(gallery_feats, dim=0)
    iterator = tqdm(enumerate(qr_loader), total=len(qr_loader), desc="Query inferencing")
    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in iterator:
        with torch.no_grad():
            #breakpoint()
            img = img.to(device)
            if segmentor is not None:
                #breakpoint()
                seg_img.to(device)
                prompt = ["the person in the image"] * seg_img.shape[0]
                mask = segmentor(seg_img, prompt)[0]
                mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
                mask = torch.sigmoid(mask)
                #breakpoint()
                threshold = 0.5
                mask[mask >= threshold] = 1.0
                mask[mask < threshold] = 0.0
                img = img #* (1.0 - mask)
            
        if cfg.MODEL.SIE_CAMERA:
            camids = camids.to(device)
        else: 
            camids = None
        if cfg.MODEL.SIE_VIEW:
            target_view = target_view.to(device)
        else: 
            target_view = None
        feat = model(img, cam_label=camids, view_label=target_view)
        new_query_feats.append(feat.detach().cpu())
        new_qpids.extend(pid)
        new_qcamids.extend(camid)
        
        img_path_list.extend(imgpath)
    
    new_query_feats = torch.cat(new_query_feats, dim=0)
    ref_feats = torch.cat([new_query_feats, gallery_feats], dim=0)
    ref_pids = new_qpids + gpids
    ref_camids = new_qcamids + gcamids
    qsize = new_query_feats.shape[0]

    #breakpoint()
    #optimizer = torch.optim.Adam(params=params, lr=0.001, weight_decay=1e-4)
    #tented_model = Tent(model, optimizer, steps=1, episodic=True)
    print("Initializing Test Time Wrapper")
    # tdisted_model = beta_TDIST(model, steps=1, device='cuda', lr=0.0001, topk=10, temp=600.0, episodic=False, lite=False, use_norm=True)
    tdisted_model = beta_TDIST_lite(**tta_args)
    if not new_query: tdisted_model.find_norms(gallery_feats, gcamids)
    else: tdisted_model.find_norms(ref_feats, ref_camids, clipsize=qsize)
    tdisted_model.initialize()
    tdisted_model.train()

    iterator = tqdm(range(new_query_feats.shape[0] // batch_size + 1), total=new_query_feats.shape[0] // batch_size + 1, desc="Adapting...")
    
    for n_iter in iterator:
        if batch_size * n_iter == new_query_feats.shape[0]: break
        #breakpoint()
        
        img_feat = new_query_feats[batch_size * n_iter:(n_iter+1) * batch_size].to(device)
        pid = new_qpids[batch_size * n_iter:(n_iter+1) * batch_size]
        camid = new_qcamids[batch_size * n_iter:(n_iter+1) * batch_size]

        if not compute_entropy: 
            feat, gfeat, loss = tdisted_model(img_feat, cam_label=camid, view_label=target_view)
            iterator.set_description(f"Batch Loss: {loss:.3f}")
        else:
            feat, gfeat = tdisted_model(img, cam_label=camid, view_label=target_view)
            #breakpoint()
            _, logits = tdisted_model.model.TTA_forward(img)
            entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
            entropies.extend(entropy.detach().cpu())
        query_feats.append(feat.cpu())
        gallery_feats = gfeat.cpu()
        qpids.extend(pid)
        qcamids.extend(camid)

        # breakpoint()
        #evaluator.update((feat, pid, camid))
        img_path_list.extend(imgpath)
        if save_sample:
            img_new = img.detach().cpu()
            img1 = img_new[0]
            save_image(img1, os.path.join(path, "sample.png"))
    
    query_feats = torch.cat(query_feats, dim=0)

    if single_query is not None:
        print(f"removing everything except query camID {single_query}")
        qf, qp, qc = [], [], []
        for i in range(query_feats.shape[0]):
            if qcamids[i] in single_query:
                qf.append(query_feats[i])
                qp.append(qpids[i])
                qc.append(qcamids[i])
        query_feats = torch.stack(qf)
        qpids = qp
        qcamids = qc
        print(f"New query size: {query_feats.shape[0]} vs Old query size: {num_query}")
        num_query = query_feats.shape[0]

    feats = torch.cat([query_feats, gallery_feats], dim=0)
    evaluator.feats = feats
    evaluator.pids = qpids + gpids
    evaluator.camids = qcamids + gcamids
    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_{suffix}.pth", pickle_protocol=4)
        torch.save(qf, f"{path}/qf_{suffix}.pth", pickle_protocol=4)
        torch.save(gf, f"{path}/gf_{suffix}.pth", pickle_protocol=4)
        with open(f"{path}/imgpaths_{suffix}.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids_{suffix}.pth", pickle_protocol=4)
        torch.save(camids, f"{path}/camids_{suffix}.pth", pickle_protocol=4)

    if compute_entropy:
        E_mean = torch.stack(entropies).mean()
        E_std = torch.stack(entropies).std()
        logger.info("Entropy mean: {:.3f}, Entropy std: {:.3f}".format(E_mean, E_std))
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]