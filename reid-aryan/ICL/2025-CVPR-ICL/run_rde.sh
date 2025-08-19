# #!/bin/bash
# root_dir=/export/livia/home/vision/Ashukla/aryan/reid/Data
# tau=0.015 
# margin=0.1
# noisy_rate=0.0  #0.0 0.2 0.5 0.8
# select_ratio=0.3
# loss=TAL
# DATASET_NAME=RSTPReid
# # CUHK-PEDES ICFG-PEDES RSTPReid UFine6926

# noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
# CUDA_VISIBLE_DEVICES=0 \
#     python train.py \
#     --noisy_rate $noisy_rate \
#     --noisy_file $noisy_file \
#     --name RDE \
#     --img_aug \
#     --txt_aug \
#     --batch_size 300 \
#     --select_ratio $select_ratio \
#     --tau $tau \
#     --root_dir $root_dir \
#     --output_dir run_logs \
#     --margin $margin \
#     --dataset_name $DATASET_NAME \
#     --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}+aug+pre \
#     --num_epoch 60 \
#     --wandb_run_id 'training_RSTPReid_001' \
#     --wandb_project 'RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+pre' \
#     --distributed \
#     --resume \
#     --resume_ckpt_file './run_logs/RSTPReid/20250807_135550_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+pre/best.pth'
    
#     # --text_length 168  # for UFine6926
#     # --text_length 77 


#!/bin/bash
root_dir=/export/livia/home/vision/Ashukla/aryan/reid/Data
tau=0.015 
margin=0.1
noisy_rate=0.0  #0.0 0.2 0.5 0.8
select_ratio=0.3
loss=TAL
DATASET_NAME=CUHK-PEDES
# CUHK-PEDES ICFG-PEDES RSTPReid UFine6926

noisy_file=./noiseindex/${DATASET_NAME}_${noisy_rate}.npy
CUDA_VISIBLE_DEVICES=2 \
    python train.py \
    --noisy_rate $noisy_rate \
    --noisy_file $noisy_file \
    --name RDE \
    --img_aug \
    --txt_aug \
    --batch_size 300 \
    --select_ratio $select_ratio \
    --tau $tau \
    --root_dir $root_dir \
    --output_dir run_logs \
    --margin $margin \
    --dataset_name $DATASET_NAME \
    --loss_names ${loss}+sr${select_ratio}_tau${tau}_margin${margin}_n${noisy_rate}+aug+pre \
    --num_epoch 60 \
    --wandb_run_id 'training_CUHK-PEDES_001' \
    --wandb_project 'RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+pre' \
    --distributed \
    --resume \
    --resume_ckpt_file './run_logs/CUHK-PEDES/20250807_135525_RDE_TAL+sr0.3_tau0.015_margin0.1_n0.0+aug+pre/best.pth'
    
    # --text_length 168  # for UFine6926
    # --text_length 77 
