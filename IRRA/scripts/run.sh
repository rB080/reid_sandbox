CUDA_VISIBLE_DEVICES=1 python train.py \
	--name iira_rstp \
	--img_aug \
	--batch_size 64 \
	--MLM \
	--loss_names 'sdm+mlm+id' \
	--dataset_name 'RSTPReid' \
	--root_dir '/export/livia/home/vision/Rbhattacharya/work/data' \
	--num_epoch 60
