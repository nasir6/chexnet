import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--architecture', default='DenseNet121', 
        help='model arch, opts: [DenseNet121, DenseNet169, DenseNet201]'
    )
    parser.add_argument('--data_root', default='database/xrays', help='dataset directory path')
    parser.add_argument('--file_train', default='new_split/train_sup.txt', help='file containing supervised train list')
    parser.add_argument('--file_train_unsup', default='new_split/train_unsup.txt', help='file containing unsupervised train list')
    parser.add_argument('--file_val', default='new_split/val_list.txt', help='file containing val list')
    parser.add_argument('--file_test', default='new_split/test_list.txt', help='file containing test list')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--save_dir', default='checkpoints', help='path to save checkpoint')
    parser.add_argument('--resize', type=int, default=256, help='testing resize')
    parser.add_argument('--crop_resize', type=int, default=224, help='training random crop resize')
    parser.add_argument('--num_classes', type=int, default=14, help='number of classes in the dataset')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataloading')
    parser.add_argument('--batch_size', type=int, default=16, help='dataset batch size')
    parser.add_argument('--unsup_batch_size', type=int, default=80, help='dataset batch size')
    parser.add_argument('--uda_temp', type=int, default=1, help='tmp factor in uda loss')
    parser.add_argument('--epochs', type=int, default=100, help='max number of epochs')
    parser.add_argument('--unsup_ratio', type=float, default=10.0, help='uda loss factor in total loss')
    
    parser.add_argument('--uda', action='store_true', help='train with uda settings')

    parser.add_argument('--test_only', action='store_true', help='test only from pretrained model')

    parser.add_argument('--pretrained', type=bool, default=True, help='load imagenet pretrained model')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train classifier')
    opt = parser.parse_args()
    return opt