import os
import torch
import torch.nn as nn
from config import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_clipreid import make_model
from model.clipseg import CLIPDensePredT
from processor.processor_clipreid_stage2 import do_inference, save_features, do_ttadapt
from utils.logger import setup_logger
from tent.tent import *


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

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)
    num_classes = 1041
    #breakpoint()
    #camera_num = 5
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)

    # segmentor = CLIPDensePredT(version='ViT-B/16', reduce_dim=64)
    # segmentor.eval()
    # segmentor.load_state_dict(torch.load('/export/livia/home/vision/Rbhattacharya/work/clipseg/weights/clipseg_weights/rd64-uni.pth', map_location=torch.device('cpu')), strict=False)
    segmentor = None
    
    model = configure_model(model)
    params, param_names = collect_params(model)
    #breakpoint()
    optimizer = torch.optim.Adam(params=params, lr=0.001, weight_decay=1e-4)
    tented_model = Tent(model, optimizer, steps=5, episodic=True)

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
       do_inference(cfg,
                 tented_model,
                 val_loader,
                 num_query, segmentor=segmentor, path="/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/check_features")
    #    do_inference_camidwise(cfg,
    #              model,
    #              val_loader,
    #              num_query)
        #save_features(cfg, model, val_loader, num_query, "/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test")


