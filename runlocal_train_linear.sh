
cd moco
python eval_linear.py \
    --arch moco_resnet18 \
    --weights ./HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --train_file ../dataset/imagenet100_train_clean_filelist_0.01_sd42.txt \
    --val_file ../dataset/imagenet100_val_clean_filelist.txt \
    --load_cache \
    --detect_trigger_channels \
    --channel_num 1 \
    --batch-size 4 \
    --num_views 64 \