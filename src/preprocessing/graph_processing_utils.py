# libraries
import pandas as pd 
import numpy as np
import torch
import itertools
from scipy import sparse
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch_geometric.data import Data
from tqdm import tqdm

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

def RiboDataset(dataset_split, feature_folder, data_folder, model_type, transform, out_folder):
    # load datasets train and test
    if dataset_split == 'train':
        # codon ss features 
        df_codon_ss = pd.read_pickle(feature_folder + 'Train_RNA_SS.pkl')

        dataset_df = pd.read_csv(data_folder + 'train.csv')
        transcripts = dataset_df['transcript'].tolist()
        codon_sequences = dataset_df['sequence'].tolist()
        # dict from 2 lists
        tr_2_codon = dict(zip(transcripts, codon_sequences))

        df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
        codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

        # norm counts train and test
        norm_counts = list(dataset_df['annotations'])

    elif dataset_split == 'test':
        # codon ss features 
        df_codon_ss = pd.read_pickle(feature_folder + 'Test_RNA_SS.pkl')

        dataset_df = pd.read_csv(data_folder + 'test.csv')
        transcripts = dataset_df['transcript'].tolist()
        codon_sequences = dataset_df['sequence'].tolist()
        # dict from 2 lists
        tr_2_codon = dict(zip(transcripts, codon_sequences))

        df_codon_ss = df_codon_ss[df_codon_ss['transcript'].isin(transcripts)]
        codon_ss = df_codon_ss['codon_RNA_SS'].tolist()

        # norm counts train and test
        norm_counts = list(dataset_df['annotations'])
        cov_mod = list(dataset_df['coverage_mod'])

    for i in tqdm(range(len(codon_sequences))):
        full_adj_mat = np.asarray(codon_ss[i].todense())
        len_Seq = full_adj_mat.shape[0]

        codon_seq_sample = tr_2_codon[transcripts[i]]

        # process the codon sequence
        codon_seq_sample = codon_seq_sample[1:-1].split(',')

        # make an undirected sequence graph
        adj_mat_init = np.zeros((len_Seq, len_Seq))
        for j in range(len_Seq-1):
            adj_mat_init[j][j+1] = 1
            adj_mat_init[j+1][j] = 1
        
        # subtract ribosome neighbourhood graph from sequence graph to get three d neighbours graph
        adj_mat_3d = full_adj_mat - adj_mat_init # undirected 3d neighbours graph
        adj_mat_fin = adj_mat_3d + adj_mat_init

        # convert to sparse matrix
        adj_mat_fin = sparse.csr_matrix(adj_mat_fin)

        # convert to edge index and edge weight
        ei, ew = from_scipy_sparse_matrix(adj_mat_fin)

        # output label
        y_i = norm_counts[i]
        y_i = y_i[1:-1].split(',')
        y_i = [float(el) for el in y_i]
        y_i = slidingWindowZeroToNan(y_i, window_size=30)
        y_i = [1 + el for el in y_i]
        y_i = np.asarray(y_i)
        y_i = np.log(y_i)
        y_i = torch.from_numpy(y_i).float()

        # make a data object
        x_dict = {'codon_seq': codon_seq_sample}
        data = Data(edge_index = ei, x = x_dict, y = y_i)

        data = transform(data)

        del ei, x_dict, y_i

        torch.save(data, out_folder + dataset_split + '/sample_' + str(i) + '.pt')