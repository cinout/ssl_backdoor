cd moco
python eval_linear_debug.py \
    --arch moco_resnet18 \
    --weights ./HTBA_trigger_10_targeted_n02106550/0002/mocom0.999_contr1tau0.2_mlp_aug+_cos_b256_lr0.06_e120,160,200/checkpoint_0199.pth.tar \
    --val_file nothing \