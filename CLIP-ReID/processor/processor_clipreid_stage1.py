import logging
import os
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from torch.cuda import amp
import torch.distributed as dist
import collections
from torch.nn import functional as F
from loss.supcontrast import SupConLoss, CLIPClsLoss
from model.make_model_clipreid_test import *

def do_train_stage1(cfg,
             model,
             train_loader_stage1,
             optimizer,
             scheduler,
             local_rank):
    checkpoint_period = cfg.SOLVER.STAGE1.CHECKPOINT_PERIOD
    device = "cuda"
    epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS
    log_period = cfg.SOLVER.STAGE1.LOG_PERIOD 

    logger = logging.getLogger("transreid.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)  

    loss_meter = AverageMeter()
    scaler = amp.GradScaler()
    xent = SupConLoss(device)
    xent_cam = CLIPClsLoss(device)
    
    # train
    import time
    from datetime import timedelta
    all_start_time = time.monotonic()
    logger.info("model: {}".format(model))
    image_features = []
    labels = []
    cams = []
    with torch.no_grad():
        for n_iter, (img, vid, target_cam, target_view) in enumerate(train_loader_stage1):
            #breakpoint()
            img = img.to(device)
            target = vid.to(device)
            camid = target_cam.to(device)
            with amp.autocast(enabled=True):
                image_feature = model(img, target, get_image = True)
                for i, c, img_feat in zip(target, camid, image_feature):
                    labels.append(i)
                    cams.append(c)
                    image_features.append(img_feat.cpu())
        labels_list = torch.stack(labels, dim=0) #N
        cams_list = torch.stack(cams, dim=0) #N
        image_features_list = torch.stack(image_features, dim=0).cuda()

        batch = cfg.SOLVER.STAGE1.IMS_PER_BATCH
        num_image = labels_list.shape[0]
        i_ter = num_image // batch
    del labels, image_features, cams

    for epoch in range(1, epochs + 1):
        loss_meter.reset()
        scheduler.step(epoch)
        model.train()

        iter_list = torch.randperm(num_image).to(device)
        for i in range(i_ter+1):
            optimizer.zero_grad()
            if i != i_ter:
                b_list = iter_list[i*batch:(i+1)* batch]
            else:
                b_list = iter_list[i*batch:num_image]
            
            target = labels_list[b_list]
            camid = cams_list[b_list]
            image_features = image_features_list[b_list]
            with amp.autocast(enabled=True):
                # text_features = model(label = target, get_text = True) 
                text_features = model(label = (target, camid), get_text = True) #remove tuple
                if hasattr(model, "cam_prompt_learner") and isinstance(getattr(model, "cam_prompt_learner"), PromptLearner_background):
                    #camprompts = model.cam_prompt_learner(camid, stage2=False) 
                    #cam_text_features = model.text_encoder(camprompts, model.cam_prompt_learner.tokenized_prompts)
                    cam_text_features = text_features
                else: cam_text_features = None
            #if i == i_ter: breakpoint()
            loss_i2t = xent(image_features, text_features, target, target)
            loss_t2i = xent(text_features, image_features, target, target)

            loss = loss_i2t + loss_t2i
            if cam_text_features is not None:
                #breakpoint()
                #cam_loss = xent_cam(image_features, cam_text_features, camid) + xent_cam(cam_text_features, image_features, camid)
                cam_loss = xent(image_features, cam_text_features, camid, camid) + xent(cam_text_features, image_features, camid, camid)
                loss += cam_loss
            else: cam_loss = "N/A"
                
            scaler.scale(loss).backward()

            scaler.step(optimizer)
            scaler.update()

            loss_meter.update(loss.item(), img.shape[0])

            torch.cuda.synchronize()
            if (i + 1) % log_period == 0:
                if cam_loss != "N/A":
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, camloss: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, cam_loss.item(), scheduler._get_lr(epoch)[0]))
                else: 
                    logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, camloss: {}, Base Lr: {:.2e}"
                            .format(epoch, (i + 1), len(train_loader_stage1),
                                    loss_meter.avg, cam_loss, scheduler._get_lr(epoch)[0]))

        if epoch % checkpoint_period == 0:
            if cfg.MODEL.DIST_TRAIN:
                if dist.get_rank() == 0:
                    torch.save(model.state_dict(),
                               os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))
            else:
                torch.save(model.state_dict(),
                           os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_stage1_{}.pth'.format(epoch)))

    all_end_time = time.monotonic()
    total_time = timedelta(seconds=all_end_time - all_start_time)
    logger.info("Stage1 running time: {}".format(total_time))
