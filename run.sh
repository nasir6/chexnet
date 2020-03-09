python Main.py \
--num_workers 32 \
--lr 0.0001 \
--unsup_ratio 5 \
--batch_size 8 \
--unsup_batch_size 30 \
--data_root database/xrays \
--file_train new_split/train_sup.txt \
--file_train_unsup new_split/train_unsup.txt \
--file_val new_split/val_list.txt \
--file_test new_split/test_list.txt \
# --uda  \

# --checkpoint checkpoints/m-09032020-104523_best_auroc.pth.tar
# /media/nasir/Drive1/code/chexnet/