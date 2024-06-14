'''
model trained on only one dataset samples
'''

# libraries
from gnn_utils import trainGCN, FileRiboDataset # custom dataset and trainer
import random
from torch.nn.utils.rnn import pad_sequence
import torch_geometric
import torch_geometric.transforms as T
from torch import nn
from torchmetrics.functional import pearson_corrcoef
from torchmetrics import Metric
import argparse
from sklearn.model_selection import KFold
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
import argparse

edge_attr = 'None' # 'Yes' (default, will send the 2 features), 'None' (will make then None), 'Zero' (will convert them to zeros)

loss_fn = 'MAE + PCC'
features = ['embedding']

parser = argparse.ArgumentParser()
parser.add_argument('--random_walk', type=bool, default=True, help='addition of random walk embedding')
parser.add_argument('--model_algo', type=str, default='GCN_LSTM', help='model algorithm') # GCN_LSTM, GCNOnly, LSTMEmbedswGCN
parser.add_argument('--gcn_layers', type=str, default='[256, 128, 128, 64]', help='gcn layers and nodes') #
parser.add_argument('--batch_size', type=int, default=1, help='model training batch size') # 
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate of the model') # 
parser.add_argument('--algo', type=str, default='SAGE', help='convolution algorithm') # SAGE, GAT, GATv2, GINE, TF, DeepGCN (which uses TF)
parser.add_argument('--cheb_k', type=int, default=2, help='chebyshev filter K size') # SAGE, GAT, GATv2, GINE, TF, DeepGCN (which uses TF)
parser.add_argument('--dropout', type=float, default=0.1, help='dropout value') # additional rna folding features with the category of the nts
parser.add_argument('--seed', type=int, default=42, help='seed value') # random seed initialization
args = parser.parse_args()

# reproducibility
pl.seed_everything(args.seed)

random_walk = args.random_walk # True or False
model_algo = args.model_algo # GCNOnly, GCN_LSTM, Ensemble

if 'x' in args.gcn_layers:
    gcn_layers_strip = args.gcn_layers.strip('[]').split(', ')
    gcn_layers = []
    for i in gcn_layers_strip:
        if 'x' in i:
            gcn_layers += [int(i.split('x')[0])]*int(i.split('x')[1])
        else:
            gcn_layers += [int(i)]
else:
    gcn_layers = [int(i) for i in args.gcn_layers.strip('[]').split(', ')]

cheb_k = args.cheb_k
algo = args.algo 

# training arguments
tot_epochs = 200
batch_size = args.batch_size
dropout_val = args.dropout
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
random_walk_length = 32
alpha = -1
lr = args.lr

proc_data_folder = 'LiverGraphs/'

features_str = '_'.join(features)

model_name = str(model_algo) + '-' + algo + '[BS ' + str(batch_size) + ', D ' + str(dropout_val) + ', E ' + str(tot_epochs) + ', LR ' + str(lr) + ', RW ' + str(random_walk) + ', CK: ' + str(cheb_k) + ', Seed: ' + str(args.seed) +  ']' + 'L: ' + str(args.gcn_layers)

input_nums_dict = {'embedding': gcn_layers[0]}
num_inp_ft = sum([input_nums_dict[ft] for ft in features])

# start a new wandb run to track this script
wandb_logger = WandbLogger(log_model="all", project="GCN_MM", name=model_name)

# model parameters
save_loc = 'saved_models/' + model_name

train_ds = FileRiboDataset(proc_data_folder, 'train', edge_attr, shuffle=True, random_walk=random_walk)
test_ds = FileRiboDataset(proc_data_folder, 'test', edge_attr, shuffle=False, random_walk=random_walk)

print("samples in train dataset: ", len(train_ds))
print("samples in test dataset: ", len(test_ds))

# train model
model, result = trainGCN(gcn_layers, tot_epochs, batch_size, lr, save_loc, wandb_logger, train_ds, test_ds, dropout_val, num_inp_ft, algo, edge_attr, random_walk, model_algo, cheb_k)

print(1.0 - result['test'][0]['test_loss'])