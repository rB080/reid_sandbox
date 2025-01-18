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
from loss.tta_contrast import TTACont
from tqdm import tqdm
import json
from torchvision.utils import save_image

def do_ttadapt_1(cfg,
                 model,
                 prompt_learner,
                 val_loader,
                 num_query, 
                 optimizer,
                 scheduler,
                 local_rank, segmentor=None, path=None, save_sample=False,):
    
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.tta")
    logger.info('start test time adapt')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    with torch.no_grad():
        
        for n_iter, (img, seg_img, vid, target_cam, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Collecting Query Image features!"):
            #breakpoint()
            # if n_iter == num_query: break
            img = img.to(device)
            target = vid
            with amp.autocast(enabled=True):
                image_feature = model(img, get_image = True)
                for i, img_feat in zip(target, image_feature):
                    labels.append(i)
                    image_features.append(img_feat.cpu())
        #breakpoint()
        labels_list = torch.tensor(labels[:num_query]).cuda() #torch.stack(torch.tensor(labels), dim=0).cuda() #N
        image_features_list = torch.stack(image_features[:num_query], dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features
    #breakpoint()
    model = model.eval()
    text_feature_dict = {}
    for epoch in range(1, epochs + 1):
        
        if epoch > 1 and epoch % 2 == 0:
            print(f"Evaluating on epoch {epoch}")
            model.eval()
            for n_iter, (img, seg_img, vid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader), desc="Evaluating!"):
                with torch.no_grad():
                    img = img.to(device)
                    image_feature = model(img, get_image=True)
                    evaluator.update((image_feature, vid, camid))
            cmc, mAP, _, _, _, _, _ = evaluator.compute(textfeats=text_feature_dict)
            logger.info("Validation Results - Epoch: {}".format(epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            torch.cuda.empty_cache()

        loss_meter.reset()
        scheduler.step(epoch)
        iter_list = torch.randperm(num_image).to(device)
        for i in tqdm(range(i_ter+1), total=i_ter+1, desc="Iterating epoch values!"):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                prompts = prompt_learner(target) 
                text_features = model.text_encoder(prompts, prompt_learner.tokenized_prompts)
                #breakpoint()
                
                #text_features = model(label = target, get_text = True)
            #if i == i_ter: breakpoint()
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)
            
            loss = loss_i2t + loss_t2i

            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])
            for j in range(text_features.shape[0]):
                text_feature_dict[labels_list[b_list[j]].item()] = text_features[j].detach().cpu()

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(val_loader),
                                    loss_meter.avg, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            torch.save(prompt_learner.state_dict(),
                    os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ttaPL_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("TTA running time: {}".format(total_time))


def do_ttadapt_2(cfg,
                 model,
                 tta_module,
                 val_loader,
                 gallery_loader,
                 num_query, 
                 optimizer,
                 scheduler,
                 local_rank, segmentor=None, path=None, save_sample=False,):
    # write better cfg
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.tta")
    logger.info('start test time adapt')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    loss_i2t_meter = AverageMeter()
    loss_diag_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    diag = TTACont(device, temperature=1.0)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    model.eval()
    tta_module.prompt_learner.train()
    tta_module.image_adapter.train()
    tta_module.text_adapter.train()
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_i2t_meter.reset()
        loss_diag_meter.reset()
        scheduler.step(epoch)
        iterator = tqdm(enumerate(val_loader), total=len(val_loader))
        #iterator = tqdm(enumerate(gallery_loader), total=len(gallery_loader))
        for n_iter, (img, seg_img, vid, target_cam, camids, target_view, imgpath) in iterator:
            optimizer.zero_grad()
            #breakpoint()
            # if n_iter == num_query: break
            img = img.to(device)
            target = vid
            with amp.autocast(enabled=True):
                #breakpoint()
                query_mask = torch.tensor([1 if (n_iter * img.shape[0] + x) < num_query else 0 for x in range(img.shape[0])]).cuda()
                if query_mask.sum() == 0: break

                target = torch.tensor(list(target))
                image_features = model(img, get_image = True)
                prompts = tta_module.prompt_learner(target)
                text_features = model.text_encoder(prompts, tta_module.prompt_learner.tokenized_prompts)   

                z_x, z_t = tta_module.image_adapter(image_features), tta_module.text_adapter(text_features)
                z_x_norm = z_x #/ z_x.norm(dim=-1, keepdim=True)
                z_t_norm = z_t #/ z_t.norm(dim=-1, keepdim=True)
                N, C = z_x_norm.size()
                #z_x_norm = z_x_norm.reshape(N, 1, C)
                #z_t_norm = z_t_norm.reshape(N, C, 1)
                # breakpoint()
                S = torch.matmul(z_x_norm, z_t_norm.t())
                S = (S - S.mean(1).repeat(S.shape[0], 1)) / S.std(1).repeat(S.shape[0], 1)

            if query_mask.sum() > 0:
                loss_i2t = (xent(image_features, text_features, target, target, reduce=False) * query_mask).mean()
                loss_t2i = (xent(text_features, image_features, target, target, reduce=False) * query_mask).mean()
                loss_i2t_meter.update((loss_i2t + loss_t2i).item(), img.shape[0])
            else: 
                loss_i2t, loss_t2i = 0.0, 0.0
                # loss_i2t_meter.update(0.0, img.shape[0])

            loss_diag = diag(S)
            loss_diag_meter.update(loss_diag.item(), img.shape[0])

            loss = (loss_i2t + loss_t2i) * 1.0 + loss_diag * 3.0
            # breakpoint()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])   
            
            torch.cuda.synchronize()
            iterator.set_description(f"Epoch: {epoch}, Lit: {loss_i2t_meter.avg:.4f}, LD: {loss_diag_meter.avg:.4f} :- ")
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss (VL): {:.3f}, Loss (Diag): {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(val_loader),
                                    loss_i2t_meter.avg, loss_diag_meter.avg,
                                    scheduler._get_lr(epoch)[0]))

            if epoch % checkpoint_period == 0:
                torch.save(tta_module.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ttaPL_{}.pth'.format(epoch)))
        
        if epoch > 1 and epoch % 5 == 0:

            do_inference(
                cfg=cfg,
                model=model,
                tta_module=tta_module,
                val_loader=gallery_loader,
                num_query=num_query,
                segmentor=None, path="/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/train_msmt17_cam12345"
            )
    
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("TTA running time: {}".format(total_time))


def do_ttadapt_3(cfg,
                 model,
                 tta_module,
                 val_loader,
                 gallery_loader,
                 num_query, 
                 optimizer,
                 scheduler,
                 local_rank, epochs=10, segmentor=None, path=None, save_sample=False, eval=True, train_query=True):
    # write better cfg
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    # epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.tta")
    logger.info('start test time adapt')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    loss_i2t_meter = AverageMeter()
    loss_diag_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    diag = TTACont(device, temperature=1.0)
    evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    model.eval()
    tta_module.prompt_learner.train()
    tta_module.image_adapter.train()
    tta_module.text_adapter.train()
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        loss_i2t_meter.reset()
        loss_diag_meter.reset()
        scheduler.step(epoch)
        iterator = tqdm(enumerate(val_loader), total=len(val_loader))
        #iterator = tqdm(enumerate(gallery_loader), total=len(gallery_loader))

        if epoch > 1 and epoch % 1 == 0 and eval:

            do_inference(
                cfg=cfg,
                model=model,
                tta_module=tta_module,
                val_loader=gallery_loader,
                num_query=num_query,
                segmentor=None, path="/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/train_msmt17_cam12345"
            )

        for n_iter, (img, seg_img, vid, target_cam, camids, target_view, imgpath) in iterator:
            optimizer.zero_grad()
            #breakpoint()
            # if n_iter == num_query: break
            img = img.to(device)
            target = vid
            with amp.autocast(enabled=True):
                image_features = model(img, get_image = True)
                if train_query:
                    target = torch.tensor(list(target))
                    prompts = tta_module.prompt_learner(target)
                    text_features = model.text_encoder(prompts, tta_module.prompt_learner.tokenized_prompts)   
                    #breakpoint()
                else:
                    prompts_all = tta_module.prompt_learner(torch.tensor([x for x in range(0, 3060)]).cuda())
                    text_feats = []
                    for idx in range(3060 // 60):
                        text_feats.append(model.text_encoder(prompts_all[idx:idx+60], 
                                                            tta_module.prompt_learner.tokenized_prompts).detach().cpu())
                    text_features_all = torch.vstack(text_feats)
                    # breakpoint()
                    z_x, z_t = tta_module.image_adapter(image_features), text_features_all.cuda() # tta_module.text_adapter(text_features)
                    z_x_norm = z_x #/ z_x.norm(dim=-1, keepdim=True)
                    z_t_norm = z_t #/ z_t.norm(dim=-1, keepdim=True)
                    N, C = z_x_norm.size()
                    #z_x_norm = z_x_norm.reshape(N, 1, C)
                    #z_t_norm = z_t_norm.reshape(N, C, 1)
                    # breakpoint()
                    S = torch.matmul(z_x_norm, z_t_norm.t())
                    S = (S - S.mean()) / S.std()
                    # S = (S - S.mean(1).repeat(S.shape[0], 1)) / S.std(1).repeat(S.shape[0], 1)

            if train_query:
                loss_i2t = xent(image_features, text_features, target, target, reduce=True)
                loss_t2i = xent(text_features, image_features, target, target, reduce=True)
                loss_i2t_meter.update((loss_i2t + loss_t2i).item(), img.shape[0])
                loss_diag = 0.0
            else: 
                loss_i2t, loss_t2i = 0.0, 0.0
                # loss_i2t_meter.update(0.0, img.shape[0])

                loss_diag = diag(S)
                loss_diag_meter.update(loss_diag.item(), img.shape[0])

            loss = (loss_i2t + loss_t2i) * 1.0 + loss_diag * 1.0
            # breakpoint()
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])   
            
            torch.cuda.synchronize()
            iterator.set_description(f"Epoch: {epoch}, Lit: {loss_i2t_meter.avg:.4f}, LD: {loss_diag_meter.avg:.4f} :- ")
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss (VL): {:.3f}, Loss (Diag): {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(val_loader),
                                    loss_i2t_meter.avg, loss_diag_meter.avg,
                                    scheduler._get_lr(epoch)[0]))

            if epoch % checkpoint_period == 0:
                torch.save(tta_module.state_dict(),
                        os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_ttaPL_{}.pth'.format(epoch)))
        
        
    
    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("TTA running time: {}".format(total_time))

def do_inference(cfg,
                 model,
                 tta_module,
                 val_loader,
                 num_query, segmentor=None, path=None, save_sample=False):
    # breakpoint()
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

    for n_iter, (img, seg_img, pid, camid, camids, target_view, imgpath) in tqdm(enumerate(val_loader), total=len(val_loader)):
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
            feat = model(img, cam_label=camids, view_label=target_view, tta_module=tta_module)
            # feat = model(img, get_image = True)
            # feat = tta_module.image_adapter(feat)
            evaluator.update((feat, pid, camid))
            img_path_list.extend(imgpath)
            if save_sample:
                img_new = img.detach().cpu()
                img1 = img_new[0]
                save_image(img1, os.path.join(path, "sample.png"))


    cmc, mAP, distmat, pids, camids, qf, gf = evaluator.compute()
    if path is not None:
        torch.save(distmat, f"{path}/distmat_og.pth", pickle_protocol=4)
        torch.save(qf, f"{path}/qf_og.pth", pickle_protocol=4)
        torch.save(gf, f"{path}/gf_og.pth", pickle_protocol=4)
        with open(f"{path}/imgpaths.json", 'w') as f:
            f.write(json.dumps(img_path_list))
        torch.save(pids, f"{path}/pids.pth", pickle_protocol=4)
        torch.save(camids, f"{path}/camids.pth", pickle_protocol=4)


    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]


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