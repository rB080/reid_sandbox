import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
from torch.nn import functional as F
from loss.supcontrast import SupConLoss
from tqdm import tqdm
import json
from torchvision.utils import save_image
from model.make_model_clipreid import *
from loss.softmax_loss import CrossEntropyLabelSmooth
from tent.tdist2 import *
import copy

def do_train_stage2(cfg,
             model,
             center_criterion,
             train_loader_stage2,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.STAGE2.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.STAGE2.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.STAGE2.EVAL_PERIOD
    instance = cfg.DATALOADER.NUM_INSTANCE

    device = "cuda"
    epochs = cfg.SOLVER.STAGE2.MAX_EPOCHS

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  
            num_classes = model.module.num_classes
        else:
            num_classes = model.num_classes

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    xent_cam = CrossEntropyLabelSmooth(num_classes=15) #Hardcoded for MSMT17
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()

    # train
    batch = cfg.SOLVER.STAGE2.IMS_PER_BATCH
    i_ter = num_classes // batch
    left = num_classes-batch* (num_classes//batch)
    if left != 0 :
        i_ter = i_ter+1
    text_features = []
    cam_text_features = []
    with torch.no_grad():
        for i in range(i_ter):
            if i+1 != i_ter:
                l_list = torch.arange(i*batch, (i+1)* batch)
            else:
                l_list = torch.arange(i*batch, num_classes)
            with amp.autocast(enabled=True):
                text_feature = model(label = l_list, get_text = True, stage2=True)
                #_, _, text_feature = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
            text_features.append(text_feature.cpu())
        text_features = torch.cat(text_features, 0).cuda()
        # if hasattr(model, "cam_prompt_learner") and isinstance(getattr(model, "cam_prompt_learner"), PromptLearner_background):
        #     with amp.autocast(enabled=True):
        #         l_list = torch.arange(0, 15) # Hard coded for MSMT17...needs to be generalized or changed for other datasets
        #         camprompts = model.cam_prompt_learner(l_list, stage2=False) 
        #         cam_text_features = model.text_encoder(camprompts, model.cam_prompt_learner.tokenized_prompts)
        # else:
        #     cam_text_features = None
        cam_text_features = None

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()

        scheduler.step()

        model.train()
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage2):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)
            camid = target_cam.to(device)
            if cfg.MODEL.SIE_CAMERA:
                target_cam = target_cam.to(device)
            else: 
                target_cam = None
            if cfg.MODEL.SIE_VIEW:
                target_view = target_view.to(device)
            else: 
                target_view = None
            with amp.autocast(enabled=True):
                score, feat, image_features = model(x = img, label = target, cam_label=target_cam, view_label=target_view)
                logits = image_features @ text_features.t()
                if cam_text_features is not None:
                    cam_logits = image_features @ cam_text_features.t()
                #breakpoint()
                #logits = model.logit_head(image_features) 
                #breakpoint()
                #logits = image_features @ image_features.t()  #takes on img feat
                loss = loss_fn(score, feat, target, target_cam, logits)
                if cam_text_features is not None:
                    loss += 1.0 / xent_cam(cam_logits, camid)

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()

            acc = (logits.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader_stage2),
                                    loss_meter.avg, acc_meter.avg, scheduler.get_lr()[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if cfg.MODEL.DIST_TRAIN:
            pass
        else:
            logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader_stage2.batch_size / time_per_batch))

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
                for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in enumerate(val_loader):
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
                 num_query, segmentor=None, path=None, suffix='og', save_sample=False, camera_normalize=False, compute_entropy=False):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=False)#cfg.TEST.FEAT_NORM)

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
    entropies = []
    cam_entropies = []
    iterator = tqdm(enumerate(val_loader), total=len(val_loader))
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
            #breakpoint()
            # feat = model(x=img, text=None, batch_size=img.shape[0])
            # feat = torch.matmul(feat, featt.T)
            # feat = model(text="A photo of a person", batch_size=img.shape[0]) 
            if not compute_entropy: feat = model(img, cam_label=camids, view_label=target_view)
            else:
                #breakpoint()
                if hasattr(model, "cam_prompt_learner"):
                    #breakpoint()
                    feat, sim, camsim = model.TTA_forward(img)
                    cam_entropy = -(camsim * torch.log(camsim)).sum(dim=1)
                    cam_entropies.extend(cam_entropy.detach().cpu())
                else:
                    feat, sim = model.TTA_forward(img)
                entropy = -(sim * torch.log(sim)).sum(dim=1)
                entropies.extend(entropy.detach().cpu())
                if hasattr(model, "cam_prompt_learner"):
                    iterator.set_description(f"Entropy: {torch.vstack(entropies).mean().item():.3f}, Camentropy: {torch.vstack(cam_entropies).mean().item():.3f}")
                else:
                    iterator.set_description(f"Entropy: {torch.vstack(entropies).mean().item():.3f}")

            if camera_normalize:
                for cid in camid:
                    if cid in list(cam_dict.keys()):
                        cam_dict[cid].append(feat.detach().cpu())
                    else:
                        cam_dict[cid] = [feat.detach().cpu()]
            # breakpoint()
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
        feats = (feats - means) / stds
        evaluator.feats = feats
        

    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_{suffix}.pth", pickle_protocol=4)
        torch.save(qf, f"{path}/qf_{suffix}.pth", pickle_protocol=4)
        torch.save(gf, f"{path}/gf_{suffix}.pth", pickle_protocol=4)
        with open(f"{path}/imgpaths.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids.pth", pickle_protocol=4)
        torch.save(camids, f"{path}/camids.pth", pickle_protocol=4)


    if compute_entropy:
        E_mean = torch.stack(entropies).mean()
        E_std = torch.stack(entropies).std()
        logger.info("Entropy mean: {:.3f}, Entropy std: {:.3f}".format(E_mean, E_std))
        if hasattr(model, "cam_prompt_learner"):
            E_mean = torch.stack(cam_entropies).mean()
            E_std = torch.stack(cam_entropies).std()
            logger.info("Cam Entropy mean: {:.3f}, Cam Entropy std: {:.3f}".format(E_mean, E_std))
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
        feat = model(img, cam_label=camids, view_label=target_view, tta=False)
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
            feat = model(img, cam_label=camids, view_label=target_view, tta=False)
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
    tdisted_model = beta_TDIST(model, **tta_args)
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


def do_tta_inference_check_ranges(cfg,
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
    og_model = copy.deepcopy(model)

    
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
        feat = model(img, cam_label=camids, view_label=target_view, tta=False)
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
            feat = model(img, cam_label=camids, view_label=target_view, tta=False)
            new_query_feats.append(feat.detach().cpu())
            new_qpids.extend(pid)
            new_qcamids.extend(camid)
            
            img_path_list.extend(imgpath)
        
        new_query_feats = torch.cat(new_query_feats, dim=0)
        ref_feats = torch.cat([new_query_feats, gallery_feats], dim=0)
        ref_pids = new_qpids + gpids
        ref_camids = new_qcamids + gcamids
        qsize = new_query_feats.shape[0]


    for argidx, tta_argset in enumerate(tta_args):

        print(f"Using TTA Args {argidx+1} or {len(tta_args)}, Parameters: {tta_argset}")
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=False)#cfg.TEST.FEAT_NORM)
        evaluator.reset()
        query_feats, qpids, qcamids = [], [], []
        model = copy.deepcopy(og_model)
        if device:
            if torch.cuda.device_count() > 1:
                print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
                model = nn.DataParallel(model)
            model.to(device)
        model = configure_model(model)
        params, param_names = collect_params(model)
        #breakpoint()
        #optimizer = torch.optim.Adam(params=params, lr=0.001, weight_decay=1e-4)
        #tented_model = Tent(model, optimizer, steps=1, episodic=True)
        print("Initializing Test Time Wrapper")
        # tdisted_model = beta_TDIST(model, steps=1, device='cuda', lr=0.0001, topk=10, temp=600.0, episodic=False, lite=False, use_norm=True)
        tdisted_model = beta_TDIST(model, **tta_argset)
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



# def do_tta_separate_inference(cfg,
#                  model,
#                  gal_loader,
#                  qr_loader,
#                  num_query, segmentor=None, path=None, save_sample=False, camera_normalize=False, compute_entropy=False):
#     # breakpoint()
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=False)#cfg.TEST.FEAT_NORM)
#     evaluator.reset()
#     query_feats = []
#     gallery_feats = []

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)
#         if segmentor is not None: segmentor.to(device)

#     model.eval()
    
#     img_path_list = []
#     cam_dict = {}
#     entropies = []

#     for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(gal_loader), total=len(gal_loader)):
#         with torch.no_grad():
#             #breakpoint()
#             img = img.to(device)
            
#             if cfg.MODEL.SIE_CAMERA:
#                 camids = camids.to(device)
#             else: 
#                 camids = None
#             if cfg.MODEL.SIE_VIEW:
#                 target_view = target_view.to(device)
#             else: 
#                 target_view = None
            
#             feat = model.model(img, cam_label=camid, view_label=target_view)
            
#             gallery_feats.append(feat.cpu())

#     for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(qr_loader), total=len(qr_loader)):
#         with torch.no_grad():
#             #breakpoint()
#             img = img.to(device)
#             if segmentor is not None:
#                 #breakpoint()
#                 seg_img.to(device)
#                 prompt = ["the person in the image"] * seg_img.shape[0]
#                 mask = segmentor(seg_img, prompt)[0]
#                 mask = F.interpolate(mask, (img.shape[2], img.shape[3]))
#                 mask = torch.sigmoid(mask)
#                 #breakpoint()
#                 threshold = 0.5
#                 mask[mask >= threshold] = 1.0
#                 mask[mask < threshold] = 0.0
#                 img = img #* (1.0 - mask)
                
#             if cfg.MODEL.SIE_CAMERA:
#                 camids = camids.to(device)
#             else: 
#                 camids = None
#             if cfg.MODEL.SIE_VIEW:
#                 target_view = target_view.to(device)
#             else: 
#                 target_view = None
#             #breakpoint()
#             # feat = model(x=img, text=None, batch_size=img.shape[0])
#             # feat = torch.matmul(feat, featt.T)
#             # feat = model(text="A photo of a person", batch_size=img.shape[0]) 
#             #print(imgpath[0])
#             if not compute_entropy: feat, gfeat = model(img, cam_label=camid, view_label=target_view)
#             else:
#                 feat, gfeat = model(img, cam_label=camid, view_label=target_view)
#                 #breakpoint()
#                 _, logits = model.model.TTA_forward(img)
#                 entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)
#                 entropies.extend(entropy.detach().cpu())
#             query_feats.append(feat.cpu())
#             gallery_feats = gfeat.cpu()

#             # breakpoint()
#             #evaluator.update((feat, pid, camid))
#             img_path_list.extend(imgpath)
#             if save_sample:
#                 img_new = img.detach().cpu()
#                 img1 = img_new[0]
#                 save_image(img1, os.path.join(path, "sample.png"))
    
#     query_feats = torch.cat(query_feats, dim=0)
#     feats = torch.cat([query_feats, gallery_feats], dim=0)
#     evaluator.feats = feats
#     evaluator.pids = model.pids
#     evaluator.camids = model.camids
#     cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
#     if path is not None:
#         torch.save(distmat, f"{path}/distmat_tent2.pth", pickle_protocol=4)
#         torch.save(qf, f"{path}/qf_tent2.pth", pickle_protocol=4)
#         torch.save(gf, f"{path}/gf_tent2.pth", pickle_protocol=4)
#         with open(f"{path}/imgpaths.json", 'w') as f:
#             f.write(json.dumps(img_path_list))
#         torch.save(pids, f"{path}/pids.pth", pickle_protocol=4)
#         torch.save(camids, f"{path}/camids.pth", pickle_protocol=4)

#     if compute_entropy:
#         E_mean = torch.stack(entropies).mean()
#         E_std = torch.stack(entropies).std()
#         logger.info("Entropy mean: {:.3f}, Entropy std: {:.3f}".format(E_mean, E_std))
#     logger.info("Validation Results ")
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
#     return cmc[0], cmc[4]

# def do_inference_camidwise(cfg,
#                  model,
#                  val_loader,
#                  num_query):
#     # breakpoint()
#     device = "cuda"
#     logger = logging.getLogger("transreid.test")
#     logger.info("Enter inferencing")

#     evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)

#     evaluator.reset()

#     if device:
#         if torch.cuda.device_count() > 1:
#             print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
#             model = nn.DataParallel(model)
#         model.to(device)

#     model.eval()
#     img_path_list = []
#     feats = {}
#     for i in range(15):
#         feats[i] = []
#     for n_iter, (img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
#         with torch.no_grad():
#             #breakpoint()
#             img = img.to(device)
#             if cfg.MODEL.SIE_CAMERA:
#                 camids = camids.to(device)
#             else: 
#                 camids = None
#             if cfg.MODEL.SIE_VIEW:
#                 target_view = target_view.to(device)
#             else: 
#                 target_view = None
#             feat = model(img, cam_label=camids, view_label=target_view)
#             feats[camid[0]].append((feat.detach().cpu(), pid, camid))

#             #evaluator.update((feat, pid, camid))
#             img_path_list.extend(imgpath)

#     for key, feat in feats.items():
#         evaluator.reset()
#         for f in feat:
#             evaluator.update((f[0].to(device), f[1], f[2]))
#         print("Evaluating for cam id", key)
#         cmc, mAP, _, _, _, _, _ = evaluator.compute()
#         logger.info("Validation Results ")
#         #logger.info("mAP: {:.1%}".format(mAP))
#         logger.info("mAP: {}".format(mAP))
#         for r in [1, 5, 10]:
#             #logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[:][r - 1]))
#             logger.info("CMC curve, Rank-{}:{}".format(r, cmc[:][r - 1]))
    

def save_features(cfg,
                 model,
                 val_loader,
                 num_query,
                 filepath):
    # breakpoint()
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter inferencing")

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    feats = {}
    for i in range(15):
        feats[i] = []
    
    for n_iter, (img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
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
            feats[camid[0]].append(feat[0].detach().cpu().numpy().tolist())
            
            img_path_list.extend(imgpath)
    
    for k,v in feats.items():
        print(k, len(v))

    with open(os.path.join(filepath, "features_10cams.json"), 'w') as f:
        f.write(json.dumps(feats))