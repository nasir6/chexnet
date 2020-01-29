import os
import numpy as np
from PIL import Image
import pandas as pd 

import torch
from torch.utils.data import Dataset

#-------------------------------------------------------------------------------- 

class DatasetGenerator (Dataset):
    
    #-------------------------------------------------------------------------------- 
    
    
    def __init__ (self, data_dir, pathDatasetFile, transform):
    
        self.listImagePaths = []
        self.listImageLabels = []
        self.transform = transform
        self.num_classes = 14

        self._data_path = data_dir
        
        self._split = pathDatasetFile
        self._construct_imdb()
    
    def _construct_imdb(self):
        """Constructs the imdb."""
        # Compile the split data path
        
        data_entry_file = 'Data_Entry_2017.csv'
        split_path = os.path.join(self._data_path, self._split)
        print(f'{self._split} data path: {split_path}')
        with open(split_path, 'r') as f: file_names = f.readlines()
        split_file_names = np.array([file_name.strip().split(' ')[0] for file_name in file_names])
        df = pd.read_csv(f'{self._data_path}/{data_entry_file}')
        image_index = df.iloc[:, 0].values
        # import pdb; pdb.set_trace()

        _, split_index, _ = np.intersect1d(image_index, split_file_names, return_indices=True)
        

        # split_index = [index for index, value in enumerate(image_index) if value in split_file_names]

        labels = df.iloc[:, 1].values

        labels = [label.split('|') for label in labels]

        # np.setdiff1d((self._class_labels, np.array( ['No Finding'])))

        self._class_labels = np.setdiff1d(np.unique(np.concatenate(labels)), np.array( ['No Finding']))

        labels = np.array(labels)[split_index]
        image_index = image_index[split_index]

        self._class_ids = {v: i for i, v in enumerate(self._class_labels) if v != 'No Finding'}

        # remove No Finding

        # Construct the image db
        self._imdb = []
        for index in range(len(split_index)):
            if len(labels[index]) == 1 and labels[index][0] == 'No Finding':
                continue
            class_ids = [self._class_ids[label] for label in labels[index]]
            im_dir = os.path.join(self._data_path, 'images')
            self._imdb.append({
                'im_path': os.path.join(im_dir, image_index[index]),
                'labels': class_ids,
            })
        print(f'Number of images: {len(self._imdb)}')
        print(f'Number of classes: {len(self._class_ids)}')


    
    #-------------------------------------------------------------------------------- 
    
    def __getitem__(self, index):
        
        imagePath = self._imdb[index]['im_path']
        # self.listImagePaths[index]
        
        imageData = Image.open(imagePath).convert('RGB')

        labels = torch.tensor(self._imdb[index]['labels'])

        labels = labels.unsqueeze(0)
        imageLabel = torch.zeros(labels.size(0), self.num_classes).scatter_(1, labels, 1.).squeeze()

        # imageLabel= torch.FloatTensor(self.listImageLabels[index])
        
        if self.transform != None: imageData = self.transform(imageData)
        
        return imageData, imageLabel
        
    #-------------------------------------------------------------------------------- 
    
    def __len__(self):
        
        return len(self._imdb)
    
 #-------------------------------------------------------------------------------- 
    