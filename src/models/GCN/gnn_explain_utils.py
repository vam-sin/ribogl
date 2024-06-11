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

capr_oh_dict = {'S': 0, 'B': 1, 'M': 2, 'H': 3, 'I': 4, 'E': 5}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def makePosEnc(seq_len, num_enc):
    '''
    make positional encodings
    '''
    pos_enc = np.zeros((seq_len, num_enc))
    for i in range(seq_len):
        for j in range(num_enc):
            if j % 2 == 0:
                pos_enc[i][j] = np.sin(i / 10000 ** (j / num_enc))
            else:
                pos_enc[i][j] = np.cos(i / 10000 ** ((j-1) / num_enc))

    return pos_enc

class RiboDataset(Dataset):
    def __init__(self, dataset_split, feature_folder, data_folder, model_type, transform, edge_attr, sampler):
        super().__init__()

        self.dataset_split = dataset_split
        self.feature_folder = feature_folder
        self.data_folder = data_folder
        self.model_type = model_type
        self.transform = transform
        self.edge_attr = edge_attr
        self.sampler = sampler
        
        # cbert features
        df_cbert = pd.read_pickle(feature_folder + 'LIVER_CodonBERT.pkl')

        # codon ss features
        df_codon_ss = pd.read_pickle(feature_folder + 'LIVER_VRNA_SS.pkl')

        # load datasets train and test
        if dataset_split == 'train':
            dataset_df = pd.read_csv(data_folder + 'train_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')
            transcripts = dataset_df['transcript'].tolist()

            df_cbert = df_cbert[df_cbert['transcript'].isin(transcripts)]
            self.cbert_embeds = df_cbert['cbert_embeds'].tolist()
            cbert_embeds_tr = df_cbert['transcript'].tolist()

            df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
            self.codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

            # norm counts train and test
            self.norm_counts = [dataset_df[dataset_df['transcript'] == tr]['annotations'].values[0] for tr in cbert_embeds_tr]

        elif dataset_split == 'test':
            dataset_df = pd.read_csv(data_folder + 'test_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')
            transcripts = dataset_df['transcript'].tolist()

            df_cbert = df_cbert[df_cbert['transcript'].isin(transcripts)]
            self.cbert_embeds = df_cbert['cbert_embeds'].tolist()
            cbert_embeds_tr = df_cbert['transcript'].tolist()

            df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
            self.codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

            # norm counts train and test
            self.norm_counts = [dataset_df[dataset_df['transcript'] == tr]['annotations'].values[0] for tr in cbert_embeds_tr]

    def len(self):
        return len(self.norm_counts)
    
    def get(self, idx):

        full_adj_mat = np.asarray(self.codon_ss[idx].todense())
        len_Seq = len(self.cbert_embeds[idx])

        # make an undirected sequence graph
        if self.model_type == 'USeq':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1
            
            # convert to sparse matrix
            adj_mat_useq = sparse.csr_matrix(adj_mat_useq) # USeq

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_useq)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                ea.append([0])

        elif self.model_type == 'USeq+':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1
            
            # subtract ribosome neighbourhood graph from sequence graph to get three d neighbours graph
            adj_mat_3d = full_adj_mat - adj_mat_useq # undirected 3d neighbours graph
            adj_mat_useqPlus = adj_mat_3d + adj_mat_useq

            # convert to sparse matrix
            adj_mat_useqPlus = sparse.csr_matrix(adj_mat_useqPlus) # USeq+

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_useqPlus)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                if np.abs(ei[0][j] - ei[1][j]) == 1:
                    ea.append([0])
                else:
                    ea.append([1])

        elif self.model_type == 'DirSeq':
            # make a directed sequence graph
            adj_mat_dirseq = np.zeros((len_Seq, len_Seq))
            for j in range(1, len_Seq):
                adj_mat_dirseq[j][j-1] = 1 # A[i, j] = 1 denotes an edge from j to i

            # convert to sparse matrix
            adj_mat_dirseq = sparse.csr_matrix(adj_mat_dirseq) # DirSeq

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_dirseq)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                ea.append([0])

        elif self.model_type == 'DirSeq+':
            # make an undirected sequence graph
            adj_mat_useq = np.zeros((len_Seq, len_Seq))
            for j in range(len_Seq-1):
                adj_mat_useq[j][j+1] = 1
                adj_mat_useq[j+1][j] = 1

            # make a directed sequence graph
            adj_mat_dirseq = np.zeros((len_Seq, len_Seq))
            for j in range(1, len_Seq):
                adj_mat_dirseq[j][j-1] = 1 # A[i, j] = 1 denotes an edge from j to i

            # subtract ribosome neighbourhood graph from sequence graph to get three d neighbours graph
            adj_mat_3d = full_adj_mat - adj_mat_useq # undirected 3d neighbours graph
            adj_mat_dirseqPlus = adj_mat_3d + adj_mat_dirseq # add the sequence as well, because just 3d neighbours might make it too sparse

            # convert to sparse matrix
            adj_mat_dirseqPlus = sparse.csr_matrix(adj_mat_dirseqPlus) # DirSeq+

            # convert to edge index and edge weight
            ei, ew = from_scipy_sparse_matrix(adj_mat_dirseqPlus)

            # ea - edge attributes
            ea = []
            for j in range(len(ei[0])):
                if np.abs(ei[0][j] - ei[1][j]) == 1:
                    ea.append([0])
                else:
                    ea.append([1])

        # make positional encodings 
        pos_enc = makePosEnc(len_Seq, 32)

        # merge and then convert to torch tensor
        ft_vec = np.concatenate((self.cbert_embeds[idx], pos_enc), axis=1)
        ft_vec = torch.from_numpy(ft_vec).float()
                    
        # ft_vec = torch.from_numpy(self.cbert_embeds[idx]).float()

        # output label
        y_i = self.norm_counts[idx]
        y_i = y_i[1:-1].split(',')
        y_i = [float(el) for el in y_i]
        y_i = slidingWindowZeroToNan(y_i, window_size=30)
        y_i = [1 + el for el in y_i]
        y_i = np.asarray(y_i)
        y_i = np.log(y_i)
        y_i = torch.from_numpy(y_i).float()

        # edge attr
        ea = np.asarray(ea)
        ea = torch.from_numpy(ea).float()

        # make a data object
        if self.edge_attr:
            data = Data(edge_index = ei, x = ft_vec, y = y_i, edge_attr = ea)
            del ea
        else:
            data = Data(edge_index = ei, x = ft_vec, y = y_i)

        del ei, ft_vec, y_i

        # apply transform to data
        if self.transform:
            data = self.transform(data)

        if self.sampler:
            loader = NeighborLoader(data, num_neighbors=[5] * 2, batch_size=1)
            sample = next(iter(loader))

            return sample
        else:
            return data

def codonInt2Seq(codon_int):
    codon_seq_str = ''
    for el in codon_int:
        codon_seq_str += id_to_codon[el]

    return codon_seq_str

def caprOH(capr_str):
    # convert this to onehot encodings
    capr_oh = np.zeros((len(capr_str)))
    for j, el in enumerate(capr_str):
        capr_oh[j]= capr_oh_dict[el]

    return capr_oh

class FileRiboDataset(Dataset):
    def __init__(self, data_folder, dataset_split, edge_attr, shuffle, transforms, virtual_node, random_walk, capr, calm_bool):
        super().__init__()

        self.data_folder = data_folder
        self.dataset_split = dataset_split
        self.edge_attr = edge_attr
        self.transforms = transforms
        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # get all filenames in the folder
        self.files = os.listdir(data_folder + dataset_split)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.capr = capr

        # shuffle files
        if shuffle:
            np.random.shuffle(self.files)
        
        if capr:
            self.capr_ds = pd.read_csv('capr_feats.zip')
            # self.capr_ds = pd.read_csv('/nfs_home/craigher/data/capr_feats.zip')
            # sequence column, make the sequences, capped at length multiple of 3
            self.capr_ds['sequence'] = self.capr_ds['sequence'].apply(lambda x: x[:len(x) - len(x) % 3])

        self.calm_bool = calm_bool

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

        # add transforms
        if self.transforms:
            data = self.transforms(data)

        if self.capr:
            # get capr features
            # get transcript
            codon_nt_seq = codonInt2Seq(data.x.numpy())
            # search this in the capr dataset and get the capr feature
            capr_ft = caprOH(self.capr_ds[self.capr_ds['sequence'] == codon_nt_seq]['capr_feats'].values[0])
            capr_ft = torch.tensor(capr_ft, dtype=torch.long)

            # cut capr_ft to 3*len(data.x)
            capr_ft = capr_ft[:3*len(data.x)]

            data.capr = capr_ft
        
        if self.calm_bool:
            data.x = data.x['calm_embeds']
        else:
            data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)

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

class MaskedKLLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred = y_pred / torch.nansum(y_pred)
        y_true = y_true / torch.nansum(y_true)

        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.kl_div(y_pred_mask, y_true_mask, reduction="none")
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

class ResGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, algo, edge_attr):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.algo = algo

        if edge_attr != 'Yes':
            self.conv = TransformerConv(in_channels, out_channels, heads = 8, concat = False)
        else:
            self.conv = TransformerConv(in_channels, out_channels, heads = 8, concat = False, edge_dim = 256)

    def forward(self, x, ei, ea = None, et = None):
        x_1 = self.conv(x, ei)

        x = x + x_1 # residual connection

        return x 

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, algo, edge_attr, cheb_k):
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
            self.conv_in = GINEConv(nn.Linear(in_channels, out_channels), edge_dim = 256)
            self.conv_out = GINEConv(nn.Linear(in_channels, out_channels), edge_dim = 256)
        elif algo == 'DeepGCN':
            self.conv_in = ResGraphConv(in_channels, out_channels, algo, edge_attr)
            self.conv_out = ResGraphConv(in_channels, out_channels, algo, edge_attr)
        elif algo == 'TF':
            if edge_attr != 'Yes':
                self.conv_in = TransformerConv(in_channels, out_channels, heads = 8, concat = False)
                self.conv_out = TransformerConv(in_channels, out_channels, heads = 8, concat = False)
            else:
                self.conv_in = TransformerConv(in_channels, out_channels, heads = 8, concat = False, edge_dim = 256)
                self.conv_out = TransformerConv(in_channels, out_channels, heads = 8, concat = False, edge_dim = 256)
        elif algo == 'RGAT':
            self.conv_in = RGATConv(in_channels, out_channels, num_relations = 2, heads = 8, concat = False)
            self.conv_out = RGATConv(in_channels, out_channels, num_relations = 2, heads = 8, concat = False)
        elif algo == 'Cheb':
            self.conv_in = ChebConv(in_channels, out_channels, K = cheb_k)
            self.conv_out = ChebConv(in_channels, out_channels, K = cheb_k)

        self.model_type = model_type
        self.edge_attr = edge_attr

    def forward(self, x, ei, ea = None, et = None):
        if self.model_type == 'USeq' or self.model_type == 'USeq+':
            if self.algo == 'RGAT':
                if self.edge_attr == 'Yes':
                    x_in = self.conv_in(x, ei, et, ea)
                else:
                    x_in = self.conv_in(x, ei, et)
            else:
                if self.edge_attr == 'Yes':
                    x_in = self.conv_in(x, ei, ea)
                else:
                    x_in = self.conv_in(x, ei)

            return x_in

        elif self.model_type == 'DirSeq' or self.model_type == 'DirSeq+':
            if self.algo == 'RGAT':
                if self.edge_attr == 'Yes':
                    x_in = self.conv_in(x, ei, et, ea)
                    x_out = self.conv_out(x, ei.flip(dims=(0,)), et, ea)
                else:
                    x_in = self.conv_in(x, ei, et)
                    x_out = self.conv_out(x, ei.flip(dims=(0,)), et)
            else:
                if self.edge_attr == 'Yes':
                    x_in = self.conv_in(x, ei, ea)
                    x_out = self.conv_out(x, ei.flip(dims=(0,)), ea)
                else:
                    x_in = self.conv_in(x, ei)
                    x_out = self.conv_out(x, ei.flip(dims=(0,)))

            return x_in + x_out

class ConvModuleNonEmbed(nn.Module):
    def __init__(self, in_channels, out_channels, model_type, algo, edge_attr):
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
        if self.model_type == 'USeq' or self.model_type == 'USeq+':
            if self.edge_attr == 'Yes':
                x_in = self.conv_in(x, ei)
            else:
                x_in = self.conv_in(x, ei)

            return x_in

        elif self.model_type == 'DirSeq' or self.model_type == 'DirSeq+':
            if self.edge_attr == 'Yes':
                x_in = self.conv_in(x, ei)
                x_out = self.conv_out(x, ei.flip(dims=(0,)))
            else:
                x_in = self.conv_in(x, ei)
                x_out = self.conv_out(x, ei.flip(dims=(0,)))

            return x_in + x_out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        self.pe = torch.squeeze(self.pe, dim=1)
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class GCN_LSTM(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg):
        super().__init__()

        if random_walk == 'Yes':
            self.embedding = nn.Embedding(64, num_inp_ft - 32)
        else:
            self.embedding = nn.Embedding(64, num_inp_ft)
    
        self.gcn_layers = gcn_layers

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModuleNonEmbed(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModuleNonEmbed(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr))
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

        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.scheduler_alg = scheduler_alg

        self.test_perf_list = []
        self.test_mae_list = []
        self.test_transcript_list = []

    def forward(self, batch):
        x = batch.x

        if self.virtual_node == 'Yes':
            # set the last node to 16 (stop codon)
            x[-1] = 16
        
        ei = batch.edge_index

        # embed the codon sequence x
        x = self.embedding(x)

        if self.random_walk == 'Yes':
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

        if self.virtual_node:
            # remove last index of y_pred, and y 
            y_pred = y_pred[:-1]
            y = y[:-1]

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

        if self.scheduler_alg == 'CosineAnneal':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.scheduler_alg == 'Regular':
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

class Ensemble_GCN_LSTM(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr):
        super().__init__()

        ## GNN Model
        if random_walk and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 32 - 24)
        elif random_walk and capr == False:
            self.embedding = nn.Embedding(64, num_inp_ft - 32)
        elif random_walk == False and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 24)
        else:
            self.embedding = nn.Embedding(64, num_inp_ft)

        if capr:
            self.capr_embed = nn.Embedding(6, 8)
    
        self.gcn_layers = gcn_layers
        self.cheb_k = cheb_k
        self.abs_pos_enc = abs_pos_enc
        self.capr = capr

        if abs_pos_enc:
            if random_walk and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32 - 24)
            elif random_walk and capr == False:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32)
            elif random_walk == False and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 24)
            else:
                self.pos_enc = PositionalEncoding(num_inp_ft)

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, cheb_k))
            self.graph_norm_list.append(GraphNorm(gcn_layers[i+1]))

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

        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.scheduler_alg = scheduler_alg
        self.capr = capr

        self.perf_list = []
        self.mae_list = []
        self.y_pred_list = []
        self.y_true_list = []
        self.file_names = []
        self.test_transcript_list = []

        ### LSTM Independent Functions
        self.bilstm_embedding = nn.Embedding(64, 256)
        self.bilstm = nn.LSTM(256, 128, num_layers = 4, bidirectional=True)

        #### FFNN
        self.linear_gcn = nn.Linear(np.sum(gcn_layers), 1)
        self.linear_lstm = nn.Linear(256, 1)
        self.linear_fin = nn.Linear(2, 1)

    def forward(self, batch):
        ### GNN Model
        if self.capr:
            capr_oh = batch.capr
            # embed capr
            capr_oh = self.capr_embed(capr_oh)
            # reshape them codon wise, group 3 nts together
            capr_oh = capr_oh.reshape(-1, 3, 8)
            # concatenate over the 3 nts
            capr_oh = capr_oh.flatten(1, 2)

        x_in_gnn = batch.x

        if self.virtual_node:
            # set the last node to 16 (stop codon)
            x_in_gnn[-1] = 16
        
        ei = batch.edge_index

        # embed the codon sequence x
        x_in_gnn = self.embedding(x_in_gnn)

        if self.abs_pos_enc:
            x_in_gnn = self.pos_enc(x_in_gnn)

        if self.random_walk:
            x_in_gnn = torch.concat((x_in_gnn, batch.random_walk_pe), dim=1)

        if self.capr:
            # concatenate capr features
            x_in_gnn = torch.cat((x_in_gnn, capr_oh), dim=1)

        outputs = []

        for i in range(len(self.gcn_layers)):
            x_in_gnn = self.module_list[i](x = x_in_gnn, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x_in_gnn = x_in_gnn.reshape(x_in_gnn.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x_in_gnn = torch.mean(x_in_gnn, dim=2)

            x_in_gnn = self.graph_norm_list[i](x_in_gnn)
            x_in_gnn = self.relu(x_in_gnn)
            x_in_gnn = self.dropout(x_in_gnn)

            outputs.append(x_in_gnn)

        out_gcn = torch.cat(outputs, dim=1)

        # linear out GCN
        out_gcn = self.linear_gcn(out_gcn)
        out_gcn = out_gcn.squeeze(dim=1)

        if self.virtual_node:
            out_gcn = out_gcn[:-1]

        #### BiLSTM model
        x_in_lstm = batch.x
        
        if self.virtual_node:
            # remove last value
            x_in_lstm = x_in_lstm[:-1]

        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x_in_lstm = self.bilstm_embedding(x_in_lstm)
        x_in_lstm = torch.unsqueeze(x_in_lstm, 1)
        # x = x.permute(1, 0, 2)

        x_in_lstm, (fin_h, fin_c) = self.bilstm(x_in_lstm, (h_0, c_0))

        # linear out
        x_in_lstm = self.linear_lstm(x_in_lstm)
        
        # extra for lstm
        out_lstm = x_in_lstm.squeeze(dim=1) # (seq_len, 1)

        #### ensemble concatenation
        out_gcn = torch.unsqueeze(out_gcn, dim=1)
        out_final = torch.cat((out_gcn, out_lstm), dim=1)

        out_final = self.linear_fin(out_final)
        out_final = torch.squeeze(out_final, dim=1)
        out_final = self.softmax(out_final)

        return out_final
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        if self.virtual_node:
            # remove last index of y_pred, and y 
            # y_pred = y_pred[:-1]
            y = y[:-1]

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

        if self.scheduler_alg == 'CA':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.scheduler_alg == 'R':
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

        # convert perf to float
        perf = perf.item()
        self.perf_list.append(perf)

        self.mae_list.append(mae.item())

        y = torch.squeeze(y, dim=0)

        # append y_pred and y_true
        self.y_pred_list.append(y_pred.detach().cpu().numpy())
        self.y_true_list.append(y.detach().cpu().numpy())
        self.file_names.append(file_name)

        # if len(self.perf_list) == 1397:
        #     # convert to df
        #     df = pd.DataFrame({'perf_gnn': self.perf_list, 'mae_gnn': self.mae_list, 'y_pred_gnn': self.y_pred_list, 'y_true_gnn': self.y_true_list, 'file_names_gnn': self.file_names})
        #     # save to csv
        #     df.to_pickle('gnnOnlyDirSeqPlusTF_predictions.pkl')

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
        # x = self.initial_conv1d(x)
        # x = torch.squeeze(x, dim=1)
        # x = x.squeeze(dim=1)
        # print(x.shape)
        # x = x.permute(1, 0, 2)

        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        # # linear out
        # x = self.linear(x)
        # x = x.squeeze(dim=1)
        
        # # extra for lstm
        # out = x.squeeze(dim=1)

        # # softmax
        # out = self.softmax(out)

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

        # convert perf to float
        perf = perf.item()
        self.perf_list.append(perf)

        self.mae_list.append(mae.item())

        y = torch.squeeze(y, dim=0)

        # append y_pred and y_true
        self.y_pred_list.append(y_pred.detach().cpu().numpy())
        self.y_true_list.append(y.detach().cpu().numpy())
        self.filenames_list.append(file_name)

        # if len(self.perf_list) == 1397:
        #     # convert to df
        #     df = pd.DataFrame({'perf_lstm': self.perf_list, 'mae_lstm': self.mae_list, 'y_pred_lstm': self.y_pred_list, 'y_true_lstm': self.y_true_list, 'filenames_lstm': self.filenames_list})
        #     # save to csv
        #     df.to_pickle('lstm_predictions.pkl')

        return loss

class EnsembleLSTMEmbedswGCN(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr):
        super().__init__()

        ## GNN Model
        if random_walk and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 32 - 24)
        elif random_walk and capr == False:
            self.embedding = nn.Embedding(64, num_inp_ft - 32)
        elif random_walk == False and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 24)
        else:
            self.embedding = nn.Embedding(64, num_inp_ft)

        if capr:
            self.capr_embed = nn.Embedding(6, 8)
    
        self.gcn_layers = gcn_layers
        self.cheb_k = cheb_k
        self.abs_pos_enc = abs_pos_enc
        self.capr = capr

        if abs_pos_enc:
            if random_walk and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32 - 24)
            elif random_walk and capr == False:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32)
            elif random_walk == False and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 24)
            else:
                self.pos_enc = PositionalEncoding(num_inp_ft)

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModule(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr, cheb_k)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModule(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr, cheb_k))
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

        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.scheduler_alg = scheduler_alg
        self.capr = capr

        self.perf_list = []
        self.mae_list = []
        self.y_pred_list = []
        self.y_true_list = []
        self.file_names = []
        self.test_transcript_list = []

        ### LSTM Independent Functions
        self.bilstm_model = LSTM.load_from_checkpoint('best_model.ckpt', dropout_val=0.1, num_epochs=50, bs=1, lr=1e-4)
        self.bilstm_model.eval()

    def forward(self, batch):
        ### GNN Model
        if self.capr:
            capr_oh = batch.capr
            # embed capr
            capr_oh = self.capr_embed(capr_oh)
            # reshape them codon wise, group 3 nts together
            capr_oh = capr_oh.reshape(-1, 3, 8)
            # concatenate over the 3 nts
            capr_oh = capr_oh.flatten(1, 2)

        # get final layer embeddings from the lstm
        x_in_gnn = self.bilstm_model(batch.x)
        x_in_gnn = torch.squeeze(x_in_gnn, dim=1)

        if self.virtual_node:
            # set the last node to 16 (stop codon)
            x_in_gnn[-1] = 16
        
        ei = batch.edge_index

        if self.abs_pos_enc:
            x_in_gnn = self.pos_enc(x_in_gnn)

        if self.random_walk:
            x_in_gnn = torch.concat((x_in_gnn, batch.random_walk_pe), dim=1)

        if self.capr:
            # concatenate capr features
            x_in_gnn = torch.cat((x_in_gnn, capr_oh), dim=1)

        outputs = []

        for i in range(len(self.gcn_layers)):
            x_in_gnn = self.module_list[i](x = x_in_gnn, ei = ei)
            
            # only for GAT
            if self.algo == 'GAT' or self.algo == 'GATv2':
                x_in_gnn = x_in_gnn.reshape(x_in_gnn.shape[0], self.gcn_layers[i], 8)
                # mean over final dimension
                x_in_gnn = torch.mean(x_in_gnn, dim=2)

            x_in_gnn = self.graph_norm_list[i](x_in_gnn)
            x_in_gnn = self.relu(x_in_gnn)
            x_in_gnn = self.dropout(x_in_gnn)

            outputs.append(x_in_gnn)

        out_gcn = torch.cat(outputs, dim=1)

        # linear out GCN
        out_gcn = self.linear_gcn(out_gcn)
        out_gcn = out_gcn.squeeze(dim=1)

        if self.virtual_node:
            out_gcn = out_gcn[:-1]

        out_final = self.softmax(out_gcn)

        return out_final
    
    def _get_loss(self, batch):
        # get features and labels
        y = batch.y

        # pass through model
        y_pred = self.forward(batch)

        if self.virtual_node:
            # remove last index of y_pred, and y 
            # y_pred = y_pred[:-1]
            y = y[:-1]

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

        if self.scheduler_alg == 'CA':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.scheduler_alg == 'R':
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

        # convert perf to float
        perf = perf.item()
        self.perf_list.append(perf)

        self.mae_list.append(mae.item())

        y = torch.squeeze(y, dim=0)

        # append y_pred and y_true
        self.y_pred_list.append(y_pred.detach().cpu().numpy())
        self.y_true_list.append(y.detach().cpu().numpy())
        self.file_names.append(file_name)

        # if len(self.perf_list) == 1397:
        #     # convert to df
        #     df = pd.DataFrame({'perf_gnn': self.perf_list, 'mae_gnn': self.mae_list, 'y_pred_gnn': self.y_pred_list, 'y_true_gnn': self.y_true_list, 'file_names_gnn': self.file_names})
        #     # save to csv
        #     df.to_pickle('gnnOnlyDirSeqPlusTF_predictions.pkl')

        return loss

class GCNOnly(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr, calm_bool):
        super().__init__()

        if random_walk and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 32 - 24)
        elif random_walk and capr == False:
            self.embedding = nn.Embedding(64, num_inp_ft - 32)
        elif random_walk == False and capr:
            self.embedding = nn.Embedding(64, num_inp_ft - 24)
        else:
            self.embedding = nn.Embedding(64, num_inp_ft)

        if capr:
            self.capr_embed = nn.Embedding(6, 8)
    
        self.gcn_layers = gcn_layers
        self.cheb_k = cheb_k
        self.abs_pos_enc = abs_pos_enc
        self.capr = capr
        self.calm_bool = calm_bool

        if abs_pos_enc:
            if random_walk and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32 - 24)
            elif random_walk and capr == False:
                self.pos_enc = PositionalEncoding(num_inp_ft - 32)
            elif random_walk == False and capr:
                self.pos_enc = PositionalEncoding(num_inp_ft - 24)
            else:
                self.pos_enc = PositionalEncoding(num_inp_ft)

        if calm_bool:
            self.initial_conv1d = nn.Conv1d(1, 1, 513, padding='valid')

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

        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.scheduler_alg = scheduler_alg
        self.capr = capr

        self.perf_list = []
        self.mae_list = []
        self.y_pred_list = []
        self.y_true_list = []
        self.file_names = []
        self.test_transcript_list = []

    def forward(self, batch):
        if self.capr:
            capr_oh = batch.capr
            # embed capr
            capr_oh = self.capr_embed(capr_oh)
            # reshape them codon wise, group 3 nts together
            capr_oh = capr_oh.reshape(-1, 3, 8)
            # concatenate over the 3 nts
            capr_oh = capr_oh.flatten(1, 2)

        x = batch.x

        if self.virtual_node:
            # set the last node to 16 (stop codon)
            x[-1] = 16
        
        ei = batch.edge_index

        # embed the codon sequence x
        if self.calm_bool:
            # x = x.permute(1, 0, 2)
            # x = self.initial_conv1d(x)
            # x = torch.squeeze(x, dim=1)
            x = torch.squeeze(x, dim=0)
        else:
            x = self.embedding(x)

        if self.abs_pos_enc:
            x = self.pos_enc(x)

        if self.random_walk:
            x = torch.concat((x, batch.random_walk_pe), dim=1)

        if self.capr:
            # concatenate capr features
            x = torch.cat((x, capr_oh), dim=1)

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

        if self.virtual_node:
            # remove last index of y_pred, and y 
            y_pred = y_pred[:-1]
            y = y[:-1]

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

        if self.scheduler_alg == 'CA':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.scheduler_alg == 'R':
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

        # convert perf to float
        perf = perf.item()
        self.perf_list.append(perf)

        self.mae_list.append(mae.item())

        y = torch.squeeze(y, dim=0)

        # append y_pred and y_true
        self.y_pred_list.append(y_pred.detach().cpu().numpy())
        self.y_true_list.append(y.detach().cpu().numpy())
        self.file_names.append(file_name)

        # if len(self.perf_list) == 1397:
        #     # convert to df
        #     df = pd.DataFrame({'perf_gnn': self.perf_list, 'mae_gnn': self.mae_list, 'y_pred_gnn': self.y_pred_list, 'y_true_gnn': self.y_true_list, 'file_names_gnn': self.file_names})
        #     # save to csv
        #     df.to_pickle('gnnOnlyDirSeqPlusTF_predictions.pkl')

        return loss

class GCN_LSTMNonEmbed(L.LightningModule):                                            
    def __init__(self, gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg):
        super().__init__()

        if random_walk == 'Yes':
            self.embedding = nn.Embedding(64, num_inp_ft - 32)
        else:
            self.embedding = nn.Embedding(64, num_inp_ft)
    
        self.gcn_layers = gcn_layers

        self.module_list = nn.ModuleList()
        self.graph_norm_list = nn.ModuleList()

        self.module_list.append(ConvModuleNonEmbed(num_inp_ft, gcn_layers[0], model_type, algo, edge_attr)) 
        self.graph_norm_list.append(GraphNorm(gcn_layers[0]))
        
        for i in range(len(gcn_layers)-1):
            self.module_list.append(ConvModuleNonEmbed(gcn_layers[i], gcn_layers[i+1], model_type, algo, edge_attr))
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

        self.virtual_node = virtual_node
        self.random_walk = random_walk
        self.scheduler_alg = scheduler_alg

        self.test_perf_list = []
        self.test_mae_list = []
        self.test_transcript_list = []

    def forward(self, x, ei):
        # x = x_dict['x']
        # random_walk_pe = x_dict['rw']

        # if self.virtual_node == 'Yes':
        #     # set the last node to 16 (stop codon)
        #     x[-1] = 16

        # embed the codon sequence x
        # x = self.embedding(x)

        # if self.random_walk == 'Yes':
        #     x = torch.concat((x, random_walk_pe), dim=1)

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

        if self.virtual_node:
            # remove last index of y_pred, and y 
            y_pred = y_pred[:-1]
            y = y[:-1]

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

        if self.scheduler_alg == 'CosineAnneal':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        elif self.scheduler_alg == 'Regular':
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

def trainGCN(gcn_layers, num_epochs, bs, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, model_algo, abs_pos_enc, cheb_k, capr, calm_bool):
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
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=20),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Training
    # Check whether pretrained model exists. If yes, load it and skip training
    if model_algo == 'GCN_LSTM':
        model = GCN_LSTM(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg)
    elif model_algo == 'GCNOnly':
        model = GCNOnly(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr, calm_bool)
    elif model_algo == 'Ensemble':
        model = Ensemble_GCN_LSTM(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr)
    elif model_algo == 'LSTMEmbedswGCN':
        model = EnsembleLSTMEmbedswGCN(gcn_layers, dropout_val, num_epochs, bs, lr, num_inp_ft, model_type, algo, edge_attr, virtual_node, random_walk, scheduler_alg, abs_pos_enc, cheb_k, capr)

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
    