CUDA_VISIBLE_DEVICES=0 python Main.py \
--num_workers 32 \
--lr 0.0001 \
--unsup_ratio 1.0 \
--batch_size 16 \
--unsup_batch_size 80 \
--data_root database/xrays \
--file_train new_split/train_sup_8.txt \
--file_train_unsup new_split/train_unsup_8.txt \
--file_val new_split/val_list.txt \
--file_test new_split/test_list.txt \
--save_dir checkpoints/unsup_8 \
--rand_aug \
--uda \
--uda_temp 0.5 \
--num_classes 14 \
# --test_only \
# --checkpoint checkpoints/uda_with_10/min_loss.pth.tar \
# --checkpoint checkpoints/with_uda_with_nf/min_loss.pth.tar \
# --checkpoint checkpoints/with_uda/best_auroc.pth.tar \
# --checkpoint checkpoints/without_uda/best_auroc.pth.tar \

# dir opt: 
    # base: without any aug and without uda
    # without_uda: with rand aug and without uda
    # with_uda: with rand aug and with uda

    # best_auroc: best auroc on validation set 
    # min_loss: minimum loss on validation set 
