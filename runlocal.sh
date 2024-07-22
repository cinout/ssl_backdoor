
cd moco
python eval_detector.py \
    --arch moco_resnet18 \
    --weights ./HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True_filelist.txt \
    --num_views 16 \
    --batch_size 8 \
    --detector InterViews \
    --interview_task spectral_signature \
    --aug_type crop_plus_perspective \
    --seed 42 \
    # --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/dummy.txt \
    # --debug_print_views \

    # --seed 30 \
    # --seed 42 \

    # --aug_type basic_plus_rotation_rigid \
    # --aug_type crop_plus_perspective \
    # --aug_type perspective \

    # --train_file ../poison-generation/data/HTBA_trigger_10_targeted_n02106550/train/loc_random_loc-min_0.25_loc-max_0.75_alpha_0.00_width_50_rate_0.50_targeted_True_filelist.txt \