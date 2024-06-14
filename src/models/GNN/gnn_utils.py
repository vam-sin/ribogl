# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Dataset
import os
from transformers import Trainer
from sklearn.model_selection import train_test_split
import itertools
import os
import torch_geometric.transforms as T
import lightning as L
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.nn.norm import GraphNorm
from torch_geometric.nn.conv import SAGEConv, GATv2Conv, GINEConv, TransformerConv, RGATConv, GENConv, ChebConv
from torch_geometric.nn.models import DeepGCNLayer
from torch_geometric.sampler import NeighborSampler
from torch_geometric.loader import NeighborLoader
from torch.autograd import Variable
import math

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

class FileRiboDataset(Dataset):
    def __init__(self, data_folder, dataset_split, edge_attr, shuffle, random_walk):
        super().__init__()

        self.data_folder = data_folder
        self.dataset_split = dataset_split
        self.edge_attr = edge_attr
        self.random_walk = random_walk
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get all filenames in the folder
        self.files = os.listdir(data_folder + dataset_split)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # shuffle files
        if shuffle:
            np.random.shuffle(self.files)

    def len(self):
        return len(self.files)
    
    def get(self, idx):
        # load the file
        file_name = self.files[idx]
        data = torch.load(self.data_folder + self.dataset_split + '/' + self.files[idx])

        # # edge_attr
        if self.edge_attr == 'None':
            data.edge_attr = None
        elif self.edge_attr == 'Zero':
            data.edge_attr = torch.zeros(data.edge_index.shape[1], 2)

        # data.y, normalize by the sum of values
        data.y = data.y / torch.nansum(data.y)

        return data, file_name

class MaskedPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return 1 - cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )
    
class MaskedPearsonCorr(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())
    
class MaskedPCCL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = MaskedL1Loss()
        self.pcc_loss = MaskedPearsonLoss()

    def __call__(self, y_pred, y_true, mask):
        '''
        loss is the sum of the l1 loss and the pearson correlation coefficient loss
        '''

        l1 = self.l1_loss(y_pred, y_true, mask)
        pcc = self.pcc_loss(y_pred, y_true, mask)

        return l1 + pcc, pcc, l1

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, algo, edge_attr, dropout_val=0.1, cheb_k=2):
        super().__init__()

        self.algo = algo

        if algo == 'SAGE':
            self.conv_in = SAGEConv(in_channels, out_channels, project = True)
            self.conv_out = SAGEConv(in_channels, out_channels, project = True)
        elif algo == 'GAT':
            self.conv_in = GATConv(in_channels, out_channels, heads = 8, add_self_loops = False)
            self.conv_out = GATConv(in_channels, out_channels, heads = 8, add_self_loops = False)
        elif algo == 'GATv2':
            self.conv_in = GATv2Conv(in_channels, out_channels, heads = 8, add_self_loops = False)
            self.conv_out = GATv2Conv(in_channels, out_channels, heads = 8, add_self_loops = False)
        elif algo == 'GINE':
            self.conv_in = GINEConv(nn.Linear(in_channels, out_channels), edge_dim = 2)
            self.conv_out = GINEConv(nn.Linear(in_channels, out_channels), edge_dim = 2)
        elif algo == 'TF':
            if edge_attr != 'Yes':
                self.conv_in = TransformerConv(in_channels, out_channels, heads = 8, concat = False)
                self.conv_out = TransformerConv(in_channels, out_channels, heads = 8, concat = False)
            else:
                self.conv_in = TransformerConv(in_channels, out_channels, heads = 8, concat = False, edge_dim = 256)
                self.conv_out = TransformerConv(in_channels, out_channels, heads = 8, concat = False, edge_dim = 256)

        self.model_type = model_type
        self.edge_attr = edge_attr

    def forward(self, x, ei):
        x_in = self.conv_in(x, ei)

        return x_in

class GCN_LSTM(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, cheb_k):
        super().__init__()

        self.embedding = nn.Embedding(64, num_inp_ft - 32)
    
        self.gcn_layers = gcn_layers

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr, dropout_val, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, dropout_val, cheb_k))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

        self.dropout = nn.Dropout(dropout_val)

        self.bilstm = nn.LSTM(np.sum(gcn_layers), 128, num_layers = 4, bidirectional=True, dropout = dropout_val)

        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

        self.model_type = model_type
        self.algo = algo

    def forward(self, batch):
        x = batch.x

        ei = batch.edge_index

        # embed the codon sequence x
        x = self.embedding(x)

        x = torch.concat((x, batch.random_walk_pe), dim=1)

        outputs = []

        for i in range(len(self.gcn_layers)):
            x = self.module_list[i](x = x, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x = x.reshape(x.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x = torch.mean(x, dim=2)

            x = self.graph_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            outputs.append(x)

        out = torch.cat(outputs, dim=1)

        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        out = out.unsqueeze(1)
        out, (fin_h, fin_c) = self.bilstm(out, (h_0, c_0))

        # linear out
        out = self.linear(out)
        out = out.squeeze(dim=1)
        
        # extra for lstm
        out = out.squeeze(dim=1)

        # softmax out
        out = self.softmax(out)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # reduce lr on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        # add monitor
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):
        batch, filename = batch

        loss, perf, mae = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)
        self.log('train_mae', mae, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        batch, filename = batch
        loss, perf, mae = self._get_loss(batch)

        self.log('val_perf', perf)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return loss
    
    def test_step(self, batch):
        batch, filename = batch
        loss, perf, mae = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)
        self.log('test_mae', mae)

        self.test_perf_list.append(perf.item())
        self.test_mae_list.append(mae.item())

        return loss

class LSTM(L.LightningModule):
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        self.bilstm = nn.LSTM(256, 128, num_layers = 4, bidirectional=True)
        self.embedding = nn.Embedding(64, 256)
        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.perf_list = []
        self.mae_list = []
        self.out_tr = []
        self.y_pred_list = []
        self.y_true_list = []
        self.filenames_list = []

    def forward(self, x):
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x = self.embedding(x)
        x = torch.unsqueeze(x, dim=0)
        x = x.permute(1, 0, 2)

        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        return x
    
    def _get_loss(self, batch):
        # get features and labels
        x, y, file_name = batch

        y = y.squeeze(dim=0)

        # pass through model
        y_pred = self.forward(x)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_loss', loss)
        self.log('val_perf', perf)

        return loss
    
    def test_step(self, batch):
        x, y, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)

        return loss

class EnsembleLSTMEmbedswGCN(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, cheb_k):
        super().__init__()

        self.embedding = nn.Embedding(64, num_inp_ft - 32)
    
        self.gcn_layers = gcn_layers
        self.cheb_k = cheb_k

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(256, gcn_layers[0], model_type, algo, edge_attr, dropout_val, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, dropout_val, cheb_k))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

        self.linear_gcn = nn.Linear(np.sum(gcn_layers), 1)

        self.dropout = nn.Dropout(dropout_val)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

        self.model_type = model_type
        self.algo = algo

        ### LSTM Independent Functions
        self.bilstm_model = nn.LSTM(num_inp_ft, 128, num_layers = 4, bidirectional=True, dropout = dropout_val)

    def forward(self, batch):
        x = batch.x
        ei = batch.edge_index
        # get codon embeddings
        x = self.embedding(x)

        x = torch.concat((x, batch.random_walk_pe), dim=1)

        # get embeddings from the lstm
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        x = x.unsqueeze(1)
        x, (h_fin, c_fin) = self.bilstm_model(x, (h_0, c_0))
        x = torch.squeeze(x, dim=1)

        # use these embeddings as a feature for the gnn
        outputs = []

        for i in range(len(self.gcn_layers)):
            x = self.module_list[i](x = x, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x = x.reshape(x.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x = torch.mean(x, dim=2)

            x = self.graph_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            outputs.append(x)

        out_gcn = torch.cat(outputs, dim=1)

        # linear out GCN
        out_gcn = self.linear_gcn(out_gcn)
        out_gcn = out_gcn.squeeze(dim=1)

        out_final = self.softmax(out_gcn)

        return out_final
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # reduce lr on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        
        # add monitor
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):
        batch, file_name = batch

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)
        self.log('train_mae', mae, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        batch, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_perf', perf)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return loss
    
    def test_step(self, batch):
        batch, file_name = batch
        y = batch.y
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)
        self.log('test_mae', mae)

        return loss

class GCNOnly(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, cheb_k):
        super().__init__()

        self.embedding = nn.Embedding(64, num_inp_ft - 32)
    
        self.gcn_layers = gcn_layers
        self.cheb_k = cheb_k

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, cheb_k))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

        self.dropout = nn.Dropout(dropout_val)

        self.linear = nn.Linear(np.sum(gcn_layers), 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

        self.model_type = model_type
        self.algo = algo

    def forward(self, batch):

        x = batch.x

        ei = batch.edge_index

        # embed the codon sequence x
        x = self.embedding(x)

        x = torch.concat((x, batch.random_walk_pe), dim=1)

        outputs = []

        for i in range(len(self.gcn_layers)):
            x = self.module_list[i](x = x, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x = x.reshape(x.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x = torch.mean(x, dim=2)

            x = self.graph_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            outputs.append(x)

        out = torch.cat(outputs, dim=1)

        # linear out
        out = self.linear(out)
        out = out.squeeze(dim=1)

        # softmax out
        out = self.softmax(out)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # reduce lr on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        # add monitor
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):
        batch, file_name = batch

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)
        self.log('train_mae', mae, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        batch, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_perf', perf)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return loss
    
    def test_step(self, batch):
        batch, file_name = batch
        y = batch.y
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)
        self.log('test_mae', mae)

        return loss

class GCN_LSTMNonEmbed(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, random_walk, cheb_k):
        super().__init__()

        self.embedding = nn.Embedding(64, num_inp_ft - 32)
    
        self.gcn_layers = gcn_layers

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, cheb_k))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

        self.dropout = nn.Dropout(dropout_val)

        self.bilstm = nn.LSTM(np.sum(gcn_layers), 128, num_layers = 4, bidirectional=True)

        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

        self.model_type = model_type
        self.algo = algo

        self.random_walk = random_walk

        self.test_perf_list = []
        self.test_mae_list = []
        self.test_transcript_list = []

    def forward(self, x, ei):

        outputs = []

        for i in range(len(self.gcn_layers)):
            x = self.module_list[i](x = x, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x = x.reshape(x.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x = torch.mean(x, dim=2)
            
            x = self.graph_norm_list[i](x)
            x = self.relu(x)
            x = self.dropout(x)

            outputs.append(x)

        out = torch.cat(outputs, dim=1)

        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        out = out.unsqueeze(1)
        out, (fin_h, fin_c) = self.bilstm(out, (h_0, c_0))

        # linear out
        out = self.linear(out)
        out = out.squeeze(dim=1)
        
        # extra for lstm
        out = out.squeeze(dim=1)

        # softmax out
        out = self.softmax(out)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)

        # reduce lr on plateau
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        
        # add monitor
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):

        loss, perf, mae = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)
        self.log('train_mae', mae, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae = self._get_loss(batch)

        self.log('val_perf', perf)
        self.log('val_loss', loss)
        self.log('val_mae', mae)

        return loss
    
    def test_step(self, batch):
        loss, perf, mae = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)
        self.log('test_mae', mae)

        self.test_perf_list.append(perf.item())
        self.test_mae_list.append(mae.item())

        return loss

def trainGCN(gcn_layers, num_epochs, bs, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val, num_inp_ft, algo, edge_attr, random_walk, model_algo, cheb_k):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Training
    # Check whether pretrained model exists. If yes, load it and skip training
    if model_algo == 'GCN_LSTM':
        model = GCN_LSTM(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, algo, edge_attr, cheb_k)
    elif model_algo == 'GCNOnly':
        model = GCNOnly(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, algo, edge_attr, cheb_k)
    elif model_algo == 'LSTMEmbedswGCN':
        model = EnsembleLSTMEmbedswGCN(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, algo, edge_attr, cheb_k)

    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders = test_loader, verbose = False, ckpt_path = "best")
    result = {"test": test_result}

    # # Testing
    # # # pretrained model loading
    # model = GCNOnly.load_from_checkpoint(save_loc + '/epoch=118-step=665329.ckpt', gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr, num_inp_ft=num_inp_ft, model_type=model_type, algo=algo, edge_attr=edge_attr, virtual_node=virtual_node, random_walk=random_walk, scheduler_alg=scheduler_alg, abs_pos_enc = abs_pos_enc, cheb_k=cheb_k)
    # model.eval()

    # # test on this model
    # result = trainer.test(model, dataloaders=test_loader, verbose=False)
    # print(result)

    return model, result
    