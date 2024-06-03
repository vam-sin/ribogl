from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
import torch 
import captum
import pandas as pd
import h5py
import numpy as np
from riboclette.multimodality.GCN.gnn_explain_utils import GCNExp, GCN
from tqdm import tqdm

torch.backends.cudnn.enabled = False

# load a sample from the test folder
feature_folder = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlusNoTransform/test/'
data_folder = '/nfs_home/nallapar/final/riboclette/riboclette/models/xlnet/data/sh/'

# load test file
exp_dat = pd.read_csv('/nfs_home/nallapar/final/riboclette/riboclette/multimodality/GCN/predictions/exp_dat_testLiver.csv')

# load the model
tot_epochs = 500
batch_size = 2
dropout_val = 0.4
annot_thresh = 0.3
longZerosThresh_val = 20
percNansThresh_val = 0.05
random_walk_length = 32
alpha = -1
lr = 1e-2
model_type = 'DirSeq+'
algo = 'TF'
edge_attr = 'None'
features = ['embedding']
features_str = '_'.join(features)
loss_fn = 'MAE + PCC'
gcn_layers = [256, 128, 128, 64]
input_nums_dict = {'cbert_full': 768, 'codon_ss': 0, 'pos_enc': 32, 'embedding': 128}
num_inp_ft = sum([input_nums_dict[ft] for ft in features])

model_name = 'NormSoftmax' + model_type + '-' + algo + ' EA: ' + str(edge_attr) + ' DS: Liver' + '[' + str(annot_thresh) + ', ' + str(longZerosThresh_val) + ', ' + str(percNansThresh_val) + ', BS ' + str(batch_size) + ', D ' + str(dropout_val) + ' E ' + str(tot_epochs) + ' LR ' + str(lr) + '] F: ' + features_str + ' VN RW 32 -1 + GraphNorm ' + loss_fn
save_loc = 'saved_models/' + model_name
l_model = GCNExp.load_from_checkpoint(save_loc + '/epoch=126-step=311404.ckpt', gcn_layers=gcn_layers, dropout_val=dropout_val, num_epochs=tot_epochs, bs=batch_size, lr=lr, num_inp_ft=num_inp_ft, alpha=alpha, model_type=model_type, algo=algo, edge_attr=edge_attr)

explainer = Explainer(
    model=l_model, # get torch module from lightning module
    algorithm=CaptumExplainer(attribution_method=captum.attr.InputXGradient),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='node',
        return_type='raw',  # Model returns log probabilities.
    ),
)

part = 0

# Generate explanation for the node at index `10`:
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

total_num_sample = len(list(exp_dat['transcript']))

# dataset split into 64 parts, get start and stop of the part
start = int(part * len(total_num_sample) / 64)
stop = int((part + 1) * len(total_num_sample) / 64)

print("start: ", start)
print("stop: ", stop)

transcripts_list = list(exp_dat['transcript'])[start:stop]
genes_list = list(exp_dat['gene'])[start:stop]

part_length = len(transcripts_list)

filename = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlusNoTransform/GNN_Interpret_DH_S' + str(part) + '.h5'

# h5py dataset
out_ds = h5py.File(filename, 'w')

# make datasets in out_ds
node_attr_ds = out_ds.create_dataset(
    'node_attr',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

edge_attr_ds = out_ds.create_dataset(
    'edge_attr',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_pred_ds = out_ds.create_dataset(
    'y_pred',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_true_ds = out_ds.create_dataset(
    'y_true',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

transcript_ds = out_ds.create_dataset(
    'transcript',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=str)
)

gene_ds = out_ds.create_dataset(
    'gene',
    (len(part_length),),
    dtype=h5py.special_dtype(vlen=str)
)


# # explainability using captum
# convert to tqdm for progress bar
for sample_number in tqdm(range(start, stop)):
    file_name = feature_folder + 'sample_' + str(sample_number) + '.pt'
    # load the sample
    data = torch.load(file_name)
    data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)
    data.y = data.y / torch.nansum(data.y)
    data = data.to(device)
    data.edge_attr = None

    # get embeddings of the data.x
    data.x = l_model.embedding(data.x)

    # get the explanation
    indices = [k for k in range(0, data.y.shape[0])]

    print(sample_number, data.x.shape, data.edge_index.shape)

    edge_explain_sample = []
    node_explain_sample = []

    index = 0
    
    for index in indices:

        explanation = explainer(data.x, data.edge_index, index=index)

        # print(explanation.edge_mask.shape, explanation.node_mask.shape, batched_edge_index.shape)

        # add the edge_mask info to edge_index 
        edge_explain = torch.concat([data.edge_index, explanation.edge_mask.unsqueeze(dim=0)], dim=0)

        # flatten edge_explain
        edge_explain = edge_explain.view(-1)

        edge_explain_sample.append(edge_explain)

        node_explain = explanation.node_mask.sum(dim=1)

        node_explain_sample.append(node_explain)

    edge_explain_sample = torch.cat(edge_explain_sample, dim=0)
    # convert to 1d
    # edge_explain_sample = edge_explain_sample.view(-1)
    node_explain_sample = torch.cat(node_explain_sample, dim=0)

    # save the edge_explain_sample and node_explain_sample to the datasets
    node_attr_ds[sample_number] = node_explain_sample.detach().cpu().numpy()
    edge_attr_ds[sample_number] = edge_explain_sample.detach().cpu().numpy()


# # predictions and ground truth values
l_model.eval()

for sample_number in range(start, stop):
    print(sample_number)
    file_name = feature_folder + 'sample_' + str(sample_number) + '.pt'
    # load the sample
    data = torch.load(file_name)
    data.edge_attr = None

    # get the prediction
    data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)
    data = data.to(device)
    pred = l_model(data.x, data.edge_index)
    y_pred_ds[sample_number] = pred.detach().cpu().numpy()

    # get the truth
    data.y = data.y / torch.nansum(data.y)
    truth = data.y
    y_true_ds[sample_number] = truth.detach().cpu().numpy()

    # add the transcript
    transcript_ds[sample_number] = transcripts_list[sample_number]

    # add the gene
    gene_ds[sample_number] = genes_list[sample_number]
