'''
model trained on only one dataset samples
'''

# libraries
from bilstm_utils import trainLSTM, FileRiboDataset # custom dataset and trainer
from torch.utils.data import DataLoader
import argparse
from pytorch_lightning.loggers import WandbLogger
import argparse
import pytorch_lightning as pl

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='seed value') # additional rna folding features with the category of the nts
parser.add_argument('--dropout', type=float, default=0.1, help='seed value') # additional rna folding features with the category of the nts
args = parser.parse_args()

# reproducibility
pl.seed_everything(args.seed)

# training arguments
proc_data_folder = 'DirSeqPlusRW/'
tot_epochs = 50
batch_size = 1
dropout_val = args.dropout
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
lr = 1e-4
dilation = True 
features = ['embedding']
features_str = '_'.join(features)
model_name = 'LSTM DS: Liver ' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + ' Seed: ' + str(args.seed) + '] F: ' + features_str

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
