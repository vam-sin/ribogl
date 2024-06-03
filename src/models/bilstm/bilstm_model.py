'''
model trained on only one dataset samples
'''

# libraries
import numpy as np
import pandas as pd 
import torch
from utils import trainLSTM, FileRiboDataset # custom dataset and trainer
import random
from torch.nn.utils.rnn import pad_sequence
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
import argparse
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# training arguments
proc_data_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlusNoTransform/'
tot_epochs = 50
batch_size = 1
dropout_val = 0.1
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
lr = 1e-4
dilation = True 
features = ['cbert_ae', 'codon_ss']
features_str = '_'.join(features)
model_name = 'LSTM DS: Liver ' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] F: ' + features_str

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="GCN_MM", name=model_name)

# model parameters
save_loc = 'saved_models/' + model_name

# load datasets train and test
train_ds = FileRiboDataset(proc_data_folder, 'train', shuffle=True)
test_ds = FileRiboDataset(proc_data_folder, 'test', shuffle=False)

# create dataloaders using X and y
train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

print("samples in train dataset: ", len(train_ds))
print("samples in test dataset: ", len(test_ds))

# train model
model, result = trainLSTM(tot_epochs, batch_size, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val)

print(1.0 - result['test'][0]['test_loss'])
