CUDA_VISIBLE_DEVICES=4 python Main.py \
--num_workers 32 \
--lr 0.0001 \
--unsup_ratio 5.0 \
--batch_size 16 \
--unsup_batch_size 80 \
--data_root database/xrays \
--file_train new_split/train_sup_15.txt \
--file_train_unsup new_split/train_unsup_15.txt \
--file_val new_split/val_list.txt \
--file_test new_split/test_list.txt \
--save_dir checkpoints/uda_with_15 \
--rand_aug \
--uda \
--uda_temp 1 \
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
