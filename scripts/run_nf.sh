CUDA_VISIBLE_DEVICES=7 python Main.py \
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
--save_dir checkpoints/sup_8_nf \
--uda_temp 1 \
--iniclude_nf \
# --test_only \
# --checkpoint checkpoints/100_nf/min_loss.pth.tar \
# --checkpoint checkpoints/with_uda/best_auroc.pth.tar \
# --checkpoint checkpoints/without_uda/best_auroc.pth.tar \

# dir opt: 
    # base: without any aug and without uda
    # without_uda: with rand aug and without uda
    # with_uda: with rand aug and with uda

    # best_auroc: best auroc on validation set 
    # min_loss: minimum loss on validation set 
