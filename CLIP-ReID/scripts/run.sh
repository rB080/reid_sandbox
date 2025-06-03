# CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --mode ours --ep NE --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth' > ./ABL_5p1_NE.txt
# CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --mode ours --ep E --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth' > ./ABL_5p1_E.txt



# CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --mode ours --lite true --ep NE --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth' > ./ABL_5p1_lite_NE.txt
# CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --mode ours --lite true --ep E --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth' > ./ABL_5p1_lite_E.txt


# CUDA_VISIBLE_DEVICES=1 python test_clipreid_tent.py --mode ours --lite true --config_file configs/person/vit_clipreid_market.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_market_cam12345/ViT-B-16_60.pth'


# CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Duke_clipreid_ViT-B-16_60.pth'


# CUDA_VISIBLE_DEVICES=0 python test_tent.py --mode ours --lite true --config_file configs_transreid/Market/vit_transreid_stride.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/transreid_duke_oracle/transformer_120.pth'

#CUDA_VISIBLE_DEVICES=0 python test.py --norm true --config_file configs_transreid/Market/vit_transreid_stride.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/transreid_duke_oracle/transformer_120.pth'
#CUDA_VISIBLE_DEVICES=0 python test.py --config_file configs_transreid/MSMT17/vit_transreid_stride.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/transreid_msmt_oracle/transformer_120.pth'


# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/MSMT17_clipreid_ViT-B-16_60.pth'
# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Duke_clipreid_ViT-B-16_60.pth'
# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Market1501_clipreid_ViT-B-16_60.pth'

# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --norm true --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/MSMT17_clipreid_ViT-B-16_60.pth'
# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --norm true --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Duke_clipreid_ViT-B-16_60.pth'
# CUDA_VISIBLE_DEVICES=0 python test_clipreid.py --norm true --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/downloaded_models/Market1501_clipreid_ViT-B-16_60.pth'

# CUDA_VISIBLE_DEVICES=1 python test_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam10/ViT-B-16_60.pth'
#CUDA_VISIBLE_DEVICES=1 python tta_clipreid.py --config_file configs/person/vit_clipreid.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth'
#CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_clipreid.yml OUTPUT_DIR '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_market_cam12345'
# CUDA_VISIBLE_DEVICES=0 python train_clipreid.py --config_file configs/person/vit_clipreid_msmt.yml OUTPUT_DIR '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/msmt10_upper_bound'



CUDA_VISIBLE_DEVICES=0 python test_clipreid_tent.py --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth'
CUDA_VISIBLE_DEVICES=1 python test_clipreid.py --norm true --config_file configs/person/vit_clipreid_msmt.yml TEST.WEIGHT '/export/livia/home/vision/Rbhattacharya/work/reid_sandbox/CLIP-ReID/outputs/train_msmt17_cam12345/ViT-B-16_60.pth'