import os
import os.path as osp
from pathlib import Path
import json
import torch
import torch.nn as nn
from config_transreid import cfg
import argparse
from datasets.make_dataloader_clipreid import make_dataloader
from model.make_model_transreid import make_model
#from model.make_model_clipreid_test import make_model
from model.clipseg import CLIPDensePredT
from processor.processor import *
from utils.logger import setup_logger
from tent.tdist2 import *
from tent.tdist3 import *



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument("--mode", help="Inference mode", default='ours', type=str)
    parser.add_argument("--lite", help="Lite mode", default=False, type=bool)
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

    train_loader, train_loader_normal, val_loader, gallery_loader, query_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg, qg_separate=True)
    # num_classes = 1020 #751 #822
    if "msmt" in cfg.TEST.WEIGHT or "train1" in cfg.TEST.WEIGHT:
        num_classes = 1041
    elif "duke" in cfg.TEST.WEIGHT:
        num_classes = 702
    elif "market" in cfg.TEST.WEIGHT:
        num_classes = 751
    #breakpoint()
    #camera_num = 15
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
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
        # feat_save_path = Path(cfg.TEST.WEIGHT).parent
        # override = False
        # # saving initial inference
        # #breakpoint()
        # if not (feat_save_path / "camids.pth").is_file() or override:
        #     print("Saving initial inference results..")
        #     do_inference(cfg,
        #             model,
        #             val_loader,
        #             num_query, segmentor=None, path=feat_save_path, suffix='new', compute_entropy=True)
        
        # print("Loading data!")
        # output_dir = feat_save_path #"/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/check_features" #
        # feature_type = "new"
        # pids = torch.load(osp.join(output_dir, "pids.pth"))
        # camids = torch.load(osp.join(output_dir, "camids.pth"))
        # with open(osp.join(output_dir, "imgpaths.json"), 'r') as f:
        #     file_content = f.read()  # Read the entire content of the file as a string
        #     imgpaths = json.loads(file_content) 
        # #print(imgpaths[0])
        # Q, G = torch.load(osp.join(output_dir, f"qf_{feature_type}.pth")), torch.load(osp.join(output_dir, f"gf_{feature_type}.pth"))
        # print("Loaded successfully!")
        #breakpoint()
        
        feat_save_path = None #"/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/diagnosis2/"
        if feat_save_path is not None: os.makedirs(feat_save_path, exist_ok=True)

        tta_args = {
            'steps': 1,
            'device': 'cuda',
            'lr': 0.0001,
            'topk': 50,
            'temp': 200.0,
            'episodic': False,
            'lite': args.lite,
            'use_norm': True
        }


        if not tta_args['lite']:
            print("Running in Normal mode!")
            do_tta_inference(cfg,
                    model,
                    query_loader,
                    gallery_loader,
                    tta_args, num_query, segmentor=None, suffix=f"id14", 
                    path=feat_save_path, compute_entropy=False, new_query=True, single_query=None, mode=args.mode)
        else:
            print("Running in LITE mode!")
            do_tta_inference_lite(cfg,
                 model,
                 query_loader,
                 gallery_loader,
                 tta_args, num_query, segmentor=None, suffix=f"id14", 
                 path=feat_save_path, compute_entropy=False, new_query=True, single_query=None)
        ##################################### ABLATION STUDIES ########################################
        
        ##################################### ABLATION STUDIES ########################################
        ##################################### LITE MODE STUDIES ########################################
        # Process:
        # Alter processor script to accomodate LITE mode run...
        
        ##################################### LITE MODE STUDIES ########################################
        
    #    do_inference_camidwise(cfg,
    #              model,
    #              val_loader,
    #              num_query)
        #save_features(cfg, model, val_loader, num_query, "/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/test")


