import os
import numpy as np
import time
import sys

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

from DensenetModels import DenseNet121
from DensenetModels import DenseNet169
from DensenetModels import DenseNet201
from DatasetGenerator import DatasetGenerator
from autoaugment import XRaysPolicy
# from tqdm import tqdm
from torch.nn.functional import kl_div, softmax, log_softmax
#-------------------------------------------------------------------------------- 
def get_uda_loss(preds1, preds2):
    
    tmp = 1.0
    preds1 = softmax(preds1/tmp, dim=1).detach()
    preds2 = log_softmax(preds2/tmp, dim=1)
    
    loss_kldiv = kl_div(preds2, preds1, reduction='none')
    loss_kldiv = torch.sum(loss_kldiv, dim=1)

    return torch.mean(loss_kldiv)
    
class ChexnetTrainer ():

    #---- Train the densenet network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def train (pathDirData, pathFileTrain, pathFileVal, nnArchitecture, nnIsTrained, nnClassCount, trBatchSize, trMaxEpoch, transResize, transCrop, launchTimestamp, checkpoint):

        #-------------------- SETTINGS: NETWORK ARCHITECTURE
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        # model = torch.nn.DataParallel(model).cuda()
                
        #-------------------- SETTINGS: DATA TRANSFORMS
        
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        transformList = []
        transformList.append(transforms.RandomResizedCrop(transCrop))
        transformList.append(transforms.RandomHorizontalFlip())
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)      
        transformSequence=transforms.Compose(transformList)

        transform_only_aug = transforms.Compose([XRaysPolicy()])
        transform_with_aug = transforms.Compose([
            XRaysPolicy(),
            transforms.RandomResizedCrop(transCrop),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        #-------------------- SETTINGS: DATASET BUILDERS
        datasetTrain = DatasetGenerator(pathDirData, pathFileTrain, transform=transform_with_aug)
        datasetTrainUnsup = DatasetGenerator(pathDirData, pathFileTrain, transform=transformSequence, transform_aug=transform_only_aug)
        datasetVal =   DatasetGenerator(pathDirData, pathFileVal, transform=transformSequence)
              
        dataLoaderTrain = DataLoader(dataset=datasetTrain, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=True)
        dataLoaderUnsup = DataLoader(dataset=datasetTrainUnsup, batch_size=trBatchSize, shuffle=True,  num_workers=4, pin_memory=True)
        dataLoaderVal = DataLoader(dataset=datasetVal, batch_size=trBatchSize, shuffle=False, num_workers=4, pin_memory=True)
        
        #-------------------- SETTINGS: OPTIMIZER & SCHEDULER
        optimizer = optim.Adam (model.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)

        # optimizer = optim.SGD(
        # filter(
        #     lambda p: p.requires_grad,
        #     model.parameters()),
        # lr=0.01,
        # momentum=0.9,
        # weight_decay=5e-4)
        scheduler = ReduceLROnPlateau(optimizer, factor = 0.1, patience = 5, mode = 'min')

        #-------------------- SETTINGS: LOSS
        loss = torch.nn.BCELoss(size_average = True)
        
        #---- Load checkpoint 
        start_epoch = 0
        if checkpoint != None:
            modelCheckpoint = torch.load(checkpoint)
            model.load_state_dict(modelCheckpoint['state_dict'])
            optimizer.load_state_dict(modelCheckpoint['optimizer'])
            start_epoch = modelCheckpoint['epoch']
        
        #---- TRAIN THE NETWORK
        
        lossMIN = 100000
        max_auroc_mean = -1000
        
        for epochID in range (start_epoch, trMaxEpoch):
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampSTART = timestampDate + '-' + timestampTime
                         
            ChexnetTrainer.epochTrain (model, dataLoaderTrain, optimizer, scheduler, trMaxEpoch, nnClassCount, loss, dataLoaderUnsup)
            lossVal, losstensor, aurocMean = ChexnetTrainer.epochVal (model, dataLoaderVal, optimizer, scheduler, trMaxEpoch, nnClassCount, loss)
            
            timestampTime = time.strftime("%H%M%S")
            timestampDate = time.strftime("%d%m%Y")
            timestampEND = timestampDate + '-' + timestampTime
            
            scheduler.step(losstensor.item())
            if aurocMean > max_auroc_mean:
                max_auroc_mean = aurocMean   
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] aurocMean= ' + str(aurocMean))
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'max_suroc_mean': max_auroc_mean, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '_best_auroc.pth.tar')
            if lossVal < lossMIN:
                lossMIN = lossVal 
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'best_loss': lossMIN, 'optimizer' : optimizer.state_dict()}, 'm-' + launchTimestamp + '.pth.tar')
                print ('Epoch [' + str(epochID + 1) + '] [save] [' + timestampEND + '] loss= ' + str(lossVal))
            else:
                print ('Epoch [' + str(epochID + 1) + '] [----] [' + timestampEND + '] loss= ' + str(lossVal))
                     
    #-------------------------------------------------------------------------------- 
       
    def epochTrain (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss, unsup_loader):
        unsup_ratio = 10
        model.train()
        iter_u = iter(unsup_loader)

        for batchID, (inputs, target) in enumerate (dataLoader):
            l_data_len = len(target)
                        
            

            try:
                u_input_1, u_input_2 = next(iter_u)
            except StopIteration:
                iter_u = iter(self.unsup_loader)
                u_input_1, u_input_2 = next(iter_u)

            inputs = torch.cat([inputs, u_input_1, u_input_2])

            target = target.cuda()
            inputs = inputs.cuda()

            varInput = torch.autograd.Variable(inputs)
            varTarget = torch.autograd.Variable(target)         
            varOutput = model(inputs)
            
            lossvalue = loss(varOutput[:l_data_len], varTarget)
            
            # -------------------------uda loss
            preds_unsup = varOutput[l_data_len:]
            preds1, preds2 = torch.chunk(preds_unsup, 2)
            loss_kl_div = get_uda_loss(preds1, preds2)
            lossvalue = lossvalue + (unsup_ratio*loss_kl_div)
            #--------------------------


            optimizer.zero_grad()
            lossvalue.backward()
            optimizer.step()
            if batchID % 10 == 9:
                print(f"[{batchID:04}/{len(dataLoader)}]    loss: {lossvalue.item():0.5f}")
                #  loss kl: {unsup_ratio*loss_kl_div.item():0.5f}
            
    #-------------------------------------------------------------------------------- 
        
    def epochVal (model, dataLoader, optimizer, scheduler, epochMax, classCount, loss):
        
        model.eval ()
        
        lossVal = 0
        lossValNorm = 0
        
        losstensorMean = 0
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
        for i, (input_, target) in enumerate (dataLoader):
            with torch.no_grad():
            
                target = target.cuda()
                input_ = input_.cuda()
                outGT = torch.cat((outGT, target), 0)
                varInput = torch.autograd.Variable(input_)
                varTarget = torch.autograd.Variable(target)    
                varOutput = model(varInput)
                outPRED = torch.cat((outPRED, varOutput), 0)
                losstensor = loss(varOutput, varTarget)
                losstensorMean += losstensor
                
                lossVal += losstensor.item()
                lossValNorm += 1
                del varOutput, varTarget, varInput, target, input_
        outLoss = lossVal / lossValNorm
        losstensorMean = losstensorMean / lossValNorm
        
        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, classCount)
        aurocMean = np.array(aurocIndividual).mean()
        print ('AUROC mean ', aurocMean)

        return outLoss, losstensorMean, aurocMean
               
    #--------------------------------------------------------------------------------     
     
    #---- Computes area under ROC curve 
    #---- dataGT - ground truth data
    #---- dataPRED - predicted data
    #---- classCount - number of classes
    
    def computeAUROC (dataGT, dataPRED, classCount):
        
        outAUROC = []
        
        datanpGT = dataGT.cpu().numpy()
        datanpPRED = dataPRED.cpu().numpy()
        
        for i in range(classCount):
            outAUROC.append(roc_auc_score(datanpGT[:, i], datanpPRED[:, i]))
            
        return outAUROC
        
        
    #--------------------------------------------------------------------------------  
    
    #---- Test the trained network 
    #---- pathDirData - path to the directory that contains images
    #---- pathFileTrain - path to the file that contains image paths and label pairs (training set)
    #---- pathFileVal - path to the file that contains image path and label pairs (validation set)
    #---- nnArchitecture - model architecture 'DENSE-NET-121', 'DENSE-NET-169' or 'DENSE-NET-201'
    #---- nnIsTrained - if True, uses pre-trained version of the network (pre-trained on imagenet)
    #---- nnClassCount - number of output classes 
    #---- trBatchSize - batch size
    #---- trMaxEpoch - number of epochs
    #---- transResize - size of the image to scale down to (not used in current implementation)
    #---- transCrop - size of the cropped image 
    #---- launchTimestamp - date/time, used to assign unique name for the checkpoint file
    #---- checkpoint - if not None loads the model and continues training
    
    def test (pathDirData, pathFileTest, pathModel, nnArchitecture, nnClassCount, nnIsTrained, trBatchSize, transResize, transCrop, launchTimeStamp):   
        
        
        CLASS_NAMES = [ 'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 'Pneumonia',
                'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        
        cudnn.benchmark = True
        
        #-------------------- SETTINGS: NETWORK ARCHITECTURE, MODEL LOAD
        if nnArchitecture == 'DENSE-NET-121': model = DenseNet121(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-169': model = DenseNet169(nnClassCount, nnIsTrained).cuda()
        elif nnArchitecture == 'DENSE-NET-201': model = DenseNet201(nnClassCount, nnIsTrained).cuda()
        
        # model = torch.nn.DataParallel(model).cuda() 
        
        modelCheckpoint = torch.load(pathModel)
        model.load_state_dict(modelCheckpoint['state_dict'])

        #-------------------- SETTINGS: DATA TRANSFORMS, TEN CROPS
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        #-------------------- SETTINGS: DATASET BUILDERS
        transformList = []
        transformList.append(transforms.Resize(transResize))
        transformList.append(transforms.TenCrop(transCrop))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])))
        transformList.append(transforms.Lambda(lambda crops: torch.stack([normalize(crop) for crop in crops])))
        transformSequence=transforms.Compose(transformList)
        
        datasetTest = DatasetGenerator(pathDirData, pathFileTest, transform=transformSequence)
        dataLoaderTest = DataLoader(dataset=datasetTest, batch_size=trBatchSize, num_workers=8, shuffle=False, pin_memory=True)
        
        outGT = torch.FloatTensor().cuda()
        outPRED = torch.FloatTensor().cuda()
       
        model.eval()
        
        for i, (input, target) in enumerate(dataLoaderTest):
            with torch.no_grad():
                target = target.cuda()
                outGT = torch.cat((outGT, target), 0)
                
                bs, n_crops, c, h, w = input.size()
                
                varInput = torch.autograd.Variable(input.view(-1, c, h, w).cuda())
                
                out = model(varInput)
                outMean = out.view(bs, n_crops, -1).mean(1)
                
                outPRED = torch.cat((outPRED, outMean.data), 0)
                # del varOutput, varTarget, varInput, target, input_


        aurocIndividual = ChexnetTrainer.computeAUROC(outGT, outPRED, nnClassCount)
        aurocMean = np.array(aurocIndividual).mean()
        
        print ('AUROC mean ', aurocMean)
        
        for i in range (0, len(aurocIndividual)):
            print (CLASS_NAMES[i], ' ', aurocIndividual[i])
        
     
        return
#-------------------------------------------------------------------------------- 





