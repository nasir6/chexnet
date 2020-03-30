import os
import numpy as np
import time
import sys
import datetime
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn.functional as tfunc
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as func

from sklearn.metrics.ranking import roc_auc_score

from DensenetModels import DenseNet121, DenseNet169, DenseNet201
from DatasetGenerator import DatasetGenerator
from autoaugment import XRaysPolicy
# from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax

model_map = {
    'DenseNet121': DenseNet121,
    'DenseNet169': DenseNet169,
    'DenseNet201': DenseNet201 
}
#-------------------------------------------------------------------------------- 
    
class ChexnetTrainer ():
    def __init__(self, args):
        self.args = args

        # init model
        
        self.model = model_map[self.args.architecture](self.args.num_classes, self.args.pretrained)

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # if torch.cuda.device_count() > 1:
        #     self.model = nn.DataParallel(self.model)
        #     self.args.batch_size*=torch.cuda.device_count()
        #     self.args.unsup_batch_size*=torch.cuda.device_count()
        #     print(f"using {torch.cuda.device_count()} GPUs! with {self.args.batch_size} batch_size")
        #     # self.args.lr = 0.1 * (self.args.batch_size/256)
        #     print(f"updated learning rate {self.args.lr}")
        #     # torch.cuda.device_count()


        self.model = self.model.cuda()
            
        # 0.0001
        self.optimizer = optim.Adam (self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
        self.scheduler = ReduceLROnPlateau(self.optimizer, factor = 0.1, patience = 5, mode = 'min')
        self.criterion = torch.nn.BCELoss(reduction='mean')

        self.start_epoch = 0

        self.load_checkpoint(args.checkpoint)
        self.initDataLoaders()

    def initDataLoaders(self):
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(self.args.crop_resize))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        transform_only_aug = transforms.Compose([XRaysPolicy()])
        transform_with_aug = transforms.Compose([
            XRaysPolicy() if self.args.rand_aug else None,
            transforms.RandomResizedCrop(self.args.crop_resize),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

        test_transformSequence = transforms.Compose([
                transforms.Resize(self.args.resize),
                transforms.TenCrop(self.args.crop_resize),
                transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])) 
            ])
        
        datasetTest = DatasetGenerator(self.args.data_root, self.args.file_test, transform=test_transformSequence)
        self.dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=self.args.batch_size, num_workers=self.args.num_workers, shuffle=False )
        
        datasetTrain = DatasetGenerator(self.args.data_root, self.args.file_train, transform=transformSequence)
        if self.args.uda:
            datasetTrain = DatasetGenerator(self.args.data_root, self.args.file_train, transform=transform_with_aug)


        datasetTrainUnsup = DatasetGenerator(self.args.data_root, self.args.file_train_unsup, transform=transformSequence, transform_aug=transform_only_aug)
        datasetVal =   DatasetGenerator(self.args.data_root, self.args.file_val, transform=transformSequence)
              
        self.dataLoaderTrainSup = DataLoader(dataset=datasetTrain, batch_size=self.args.batch_size, shuffle=True,  num_workers=self.args.num_workers )
        self.dataLoaderUnsup = DataLoader(dataset=datasetTrainUnsup, batch_size=self.args.unsup_batch_size, shuffle=True,  num_workers=self.args.num_workers )
        self.dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers )
        
    def get_uda_loss(self, preds1, preds2):
    
        tmp = self.args.uda_temp
        preds1 = softmax(preds1/tmp, dim=1).detach()
        preds2 = log_softmax(preds2/tmp, dim=1)
        
        loss_kldiv = kl_div(preds2, preds1, reduction='none')
        loss_kldiv = torch.sum(loss_kldiv, dim=1)

        return torch.mean(loss_kldiv)

    def load_checkpoint(self, checkpoint):
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            self.model.load_state_dict(modelCheckpoint['state_dict'])
            self.optimizer.load_state_dict(modelCheckpoint['optimizer'])
            self.start_epoch = modelCheckpoint['epoch']
            print(f"loaded epoch {self.start_epoch} from {checkpoint} \n")
    def train (self):        

        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        max_auroc_mean = -1000
        self.iter_u = iter(self.dataLoaderUnsup)
        
        for epochID in range (self.start_epoch, self.args.epochs):
                         
            self.epochTrain(epochID)

            lossVal, losstensor, aurocMean = self.epochVal()
            
            self.scheduler.step(losstensor.item())
            if aurocMean > max_auroc_mean:
                max_auroc_mean = aurocMean
                print(f"{datetime.datetime.now()} --- \t Epoch [{epochID + 1}] [save] AUROC mean: {aurocMean:0.4f} loss: {lossVal:0.6f}")
                torch.save({'epoch': epochID + 1, 'state_dict': self.model.state_dict(), 'max_suroc_mean': max_auroc_mean, 'optimizer' : self.optimizer.state_dict()}, f'{self.args.save_dir}/best_auroc.pth.tar')
            if lossVal < lossMIN:
                lossMIN = lossVal 
                torch.save({'epoch': epochID + 1, 'state_dict': self.model.state_dict(), 'best_loss': lossMIN, 'optimizer' : self.optimizer.state_dict()}, f'{self.args.save_dir}/min_loss.pth.tar')
                print(f"{datetime.datetime.now()} --- \t Epoch [{epochID + 1}] [save] AUROC mean: {aurocMean:0.4f} loss: {lossVal:0.6f}")
            else:
                print(f"{datetime.datetime.now()} --- \t Epoch [{epochID + 1}] [----] AUROC mean: {aurocMean:0.4f} loss: {lossVal:0.6f}")
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (self, epochID):
        self.model.train()
        uda_epoch = 0
        for batchID, (inputs, target) in enumerate (self.dataLoaderTrainSup):
            target = target.cuda()
            inputs = inputs.cuda()
            varOutput = self.model(inputs)
            lossvalue = self.criterion(varOutput, target)
            if self.args.uda and epochID >= uda_epoch:
                try:
                    u_input_1, u_input_2 = next(self.iter_u)
                except StopIteration:
                    self.iter_u = iter(self.dataLoaderUnsup)
                    u_input_1, u_input_2 = next(self.iter_u)
                preds1 = self.model(u_input_1.cuda())
                preds2 = self.model(u_input_2.cuda())
                loss_kl_div = self.args.unsup_ratio*self.get_uda_loss(preds1, preds2)
                lossvalue = lossvalue + loss_kl_div

            self.optimizer.zero_grad()
            lossvalue.backward()
            self.optimizer.step()
            if batchID % 10 == 9:
                print(f"{datetime.datetime.now()} --- \t [{batchID:04}/{len(self.dataLoaderTrainSup)}] loss: {lossvalue.item():0.5f} loss UDA: {loss_kl_div.item() if self.args.uda and epochID >= uda_epoch else 0 :0.5f}")    
        
    def epochVal(self):   
        self.model.eval()
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        for i, (input_, target) in enumerate(self.dataLoaderVal):
            with torch.no_grad():
            
                target = target.cuda()
                input_ = input_.cuda()
                outGT = torch.cat((outGT, target), 0)
                varInput = torch.autograd.Variable(input_)
                varTarget = torch.autograd.Variable(target)    
                varOutput = self.model(varInput)
                outPRED = torch.cat((outPRED, varOutput), 0)
                losstensor = self.criterion(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.item()
                lossValNorm += 1
                del varOutput, varTarget, varInput, target, input_
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        aurocIndividual = self.computeAUROC(outGT, outPRED)
        aurocMean = np.array(aurocIndividual).mean()
        print (f'{datetime.datetime.now()} --- \t Val AUROC mean: {aurocMean}')

        return outLoss, losstensorMean, aurocMean
               
    #--------------------------------------------------------------------------------
    
    def computeAUROC(self, dataGT, dataPRED):
        
        outAUROC = []
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        for i in range(self.args.num_classes):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
        return outAUROC
        
    #--------------------------------------------------------------------------------  
    
    def test (self):
        CLASS_NAMES = self.dataLoaderTest.dataset._class_labels
        # [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                # 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        self.model.eval()
        for i, (input, target) in enumerate(self.dataLoaderTest):
            with torch.no_grad():
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                bs, n_crops, c, h, w = input.size()
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                out = self.model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                outPRED = torch.cat((outPRED, outMean.data), 0)
        aurocIndividual = self.computeAUROC(outGT, outPRED)
        aurocMean = np.array(aurocIndividual).mean()
        print ('Test AUROC mean ', aurocMean)
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        return





