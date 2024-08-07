python generate_bdremoved_trainset.py \
    --pred_scores_file pred_scores_trigger_None_detector_variance_aug_crop_nviews_32_bs_8_sd_50.npy \
    --output_folder poison-generation/data/HTBA_trigger_10_targeted_n02106550/train \
    --cutoff_quantile 0.9 \
