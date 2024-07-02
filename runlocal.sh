cd moco

python eval_linear.py \
    --arch moco_resnet18 \
    --weights ./HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --val_file ../dataset/imagenet100_val_clean_filelist.txt \
    --train_file ../dataset/imagenet100_train_clean_filelist.txt \
    --val_poisoned_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/val_poisoned/loc_random_loc-min_0.10_loc-max_0.90_alpha_0.00_width_50_filelist.txt \
    --resume ./HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/linear_0.01_PRE/checkpoint_0199.pth.tar/checkpoint.pth.tar \
    --evaluate \
    --load_cache \
    --batch-size 8 \
    # --eval_data <evaluation-ID> \