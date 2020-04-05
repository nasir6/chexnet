

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import os

def plot_roc(gt, preds, classes):
    
    colors = ['#8a2244', '#da8c22', '#c687d5', '#80d6f8', '#440f06', '#000075', '#000000', '#e6194B', '#f58231', '#ffe119', '#bfef45',
    '#02a92c', '#3a3075', '#3dde43', '#baa980', '#170eb8', '#f032e6', '#a9a9a9', '#fabebe', '#ffd8b1', '#fffac8', '#aaffc3', '#5b7cd4',
    '#3e319d', '#a837b2', '#400dd2', '#f8d307']
    try:
        os.makedirs('plots')
    except:
        print("plots, dir allready exists")
    # import pdb; pdb.set_trace()
    gt = gt.data.cpu().numpy()
    preds = preds.data.cpu().numpy()
    for i in range(len(classes)):
        fpr, tpr, _ = metrics.roc_curve(gt[:, i], preds[:, i])
        roc_auc = metrics.auc(fpr, tpr)

        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, colors[i], label = f'{classes[i]} AUC = {roc_auc:0.2f}')

    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(f'plots/roc_curve.pdf')

