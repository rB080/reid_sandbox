import os
import torch
import torch.nn as nn
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model, PromptLearner, TestTimeAdapter, load_clip_to_cpu
from model.clipseg import CLIPDensePredT
from processor.processor_clipreid_TTA import do_inference, do_ttadapt_1, do_ttadapt_2, do_ttadapt_3
from utils.logger import setup_logger
from solver.make_optimizer_prompt import make_optimizer_1stage, make_optimizer_2stage
from solver.scheduler_factory import create_scheduler
from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="configs/person/vit_clipreid.yml", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    train_loader, train_loader_normal, val_loader, gallery_loader, query_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, True)
    #num_classes = 998
    #breakpoint()
    #camera_num = 5
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    # segmentor = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    # segmentor.eval()
    # segmentor.load_state_dict(torch.load('/export/livia/home/vision/Rbhattacharya/work/clipseg/weights/clipseg_weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
    segmentor = None
    h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
    w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
    clip_model = load_clip_to_cpu('ViT-B-16', h_resolution, w_resolution, 16)
    clip_model.to("cuda")
    
    TTA_module = TestTimeAdapter(
        dtype=clip_model.dtype, token_embedding=clip_model.token_embedding,
        num_classes=3060, dims=512, 
    )
    TTA_module.to("cuda")
    
    optimizer = torch.optim.Adam(params=TTA_module.prompt_learner.parameters(), lr=0.0005, weight_decay=1e-4) #make_optimizer_1stage(cfg, TTA_Promptlearner)
    scheduler = create_scheduler(optimizer, num_epochs = cfg.SOLVER.STAGE1.MAX_EPOCHS, lr_min = cfg.SOLVER.STAGE1.LR_MIN, \
                        warmup_lr_init = cfg.SOLVER.STAGE1.WARMUP_LR_INIT, warmup_t = cfg.SOLVER.STAGE1.WARMUP_EPOCHS, noise_range = None)
    #breakpoint()
    
    if cfg.DATASETS.NAMES == 'VehicleID':
        for trial in range(10):
            train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
            rank_1, rank5, mAP = do_inference(cfg,
                 model,
                 val_loader,
                 num_query)
            if trial == 0:
                all_rank_1 = rank_1
                all_rank_5 = rank5
                all_mAP = mAP
            else:
                all_rank_1 = all_rank_1 + rank_1
                all_rank_5 = all_rank_5 + rank5
                all_mAP = all_mAP + mAP

            logger.info("rank_1:{}, rank_5 {} : trial : {}".format(rank_1, rank5, mAP, trial))
        logger.info("sum_rank_1:{:.1%}, sum_rank_5 {:.1%}, sum_mAP {:.1%}".format(all_rank_1.sum()/10.0, all_rank_5.sum()/10.0, all_mAP.sum()/10.0))
    else:
        do_ttadapt_3(
            cfg,
            model,
            TTA_module,
            query_loader,
            val_loader,
            num_query,
            optimizer,
            scheduler,
            local_rank=0, epochs=15, eval=False, train_query=True
        )
        # print("Loading and resuming gallery directly")
        # TTA_module.load_state_dict(torch.load("outputs/train_msmt17_cam12345/ViT-B-16_ttaPL_120.pth", map_location='cuda'))
        # iterator = tqdm(enumerate(query_loader), total=len(query_loader))
        # with torch.no_grad():
        #     prompts_all = TTA_module.prompt_learner(torch.tensor([x for x in range(0, 3060)]).cuda())
        #     text_feats = []
        #     img_feats = []
        #     pids = []
        #     for idx in range(3060 // 60):
        #         text_feats.append(model.text_encoder(prompts_all[idx:idx+60], 
        #                                             TTA_module.prompt_learner.tokenized_prompts).detach().cpu())
        #     text_features_all = torch.vstack(text_feats)
        #     for n_iter, (img, seg_img, vid, target_cam, camids, target_view, imgpath) in iterator:
        #         image_features = model(img.cuda(), get_image = True)
        #         img_feats.append(image_features.detach().cpu())
        #         pids.extend(list(vid))
        #     image_features_all = torch.vstack(img_feats)
        #     torch.save(text_features_all, "outputs/train_msmt17_cam12345/qtextfeats.pth")
        #     torch.save(image_features_all, "outputs/train_msmt17_cam12345/qimgfeats.pth")
        #     torch.save(pids, "outputs/train_msmt17_cam12345/qpids.pth")

        optimizer = torch.optim.Adam(params=TTA_module.image_adapter.parameters(), lr=0.0005, weight_decay=1e-4) #make_optimizer_1stage(cfg, TTA_Promptlearner)
        scheduler = create_scheduler(optimizer, num_epochs = 50, lr_min = 0.001, \
                                warmup_lr_init = 0.005, warmup_t = 0, noise_range = None)
        do_ttadapt_3(
            cfg,
            model,
            TTA_module,
            gallery_loader,
            val_loader,
            num_query,
            optimizer,
            scheduler,
            local_rank=0, epochs=50, train_query=False
        )
        _, _, val_loader, _, _, _, _ = make_dataloader(cfg, False)
        do_inference(cfg,
            model,
            TTA_module,
            val_loader,
            num_query, segmentor=segmentor, path="/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/train_msmt17_cam12345")


