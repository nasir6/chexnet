import pandas as pd
import numpy as np
import os

pathDirData = 'database/xrays'
# import pdb; pdb.set_trace()



def create_train_val_test_split():
    # change the path to save the new splits
    dir_path = f"{pathDirData}/new_split"


    data_entry_file = 'Data_Entry_2017.csv'
    df = pd.read_csv(f'{pathDirData}/{data_entry_file}')
    image_index = df.iloc[:, 0].values
    labels = df.iloc[:, 1].values
    # select the labels with findings
    pos_labels_inds = np.where(labels!='No Finding')[0]
    # select images with findings
    image_index = image_index[pos_labels_inds]
    # 70 % train
    num_train = int(0.7 * len(image_index))
    # 10 % val
    num_val = int(0.1 * len(image_index))
    # almost 20% test set 
    # num_test = len(image_index) - num_train - num_val

    # shuffle indexes
    image_index_shuffled = np.random.permutation(image_index)

    # pick 0:36231
    train_list = image_index_shuffled[0:num_train] 
    # pick 36231:41406

    val_list = image_index_shuffled[num_train:num_train+num_val]
    # 20 % test data pick 41406:
    test_list = image_index_shuffled[num_train+num_val:]

    # save the split to files
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    
    print(f'num of train {len(train_list)}')
    print(f'num of val {len(val_list)}')
    print(f'num of test {len(test_list)}')

    with open(f'{dir_path}/train_list.txt', 'w') as f:
        f.writelines([f"{item}\n"  for item in train_list])
    with open(f'{dir_path}/test_list.txt', 'w') as f:
        f.writelines([f"{item}\n"  for item in test_list])
    with open(f'{dir_path}/val_list.txt', 'w') as f:
        f.writelines([f"{item}\n"  for item in val_list])

def split_sup_unsup(ratios):
    # newly created split or load the old split to create unsup and sup files
    dir_path = f"{pathDirData}/new_split"
    with open(f"{dir_path}/train_list.txt", 'r') as f: file_names = f.readlines()
    file_names = np.array([file_name.strip().split(' ')[0] for file_name in file_names])
    file_names_shuffled = np.random.permutation(file_names)
    # num of sup 10% of training data
    for ratio in ratios:
    # ratio = 15
        num_sup_train = int((ratio/100) * len(file_names_shuffled))
        sup_file_list = file_names_shuffled[:num_sup_train]
        unsup_file_list = file_names_shuffled[num_sup_train:]

        print(f'num of sup {len(sup_file_list)}')
        print(f'num of unsup {len(unsup_file_list)}')
        with open(f'{dir_path}/train_sup_{ratio}.txt', 'w') as f:
            f.writelines([f"{item}\n"  for item in sup_file_list])
        with open(f'{dir_path}/train_unsup_{ratio}.txt', 'w') as f:
            f.writelines([f"{item}\n"  for item in unsup_file_list])

# create_train_val_test_split()
split_sup_unsup([2, 4, 6, 8, 12, 16])
