# import captum
from captum.attr import LayerGradientXActivation
import torch 
import matplotlib.pyplot as plt

data = torch.load('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlusNoTransform/test/sample_127.pt')

data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)
data.y = data.y / torch.nansum(data.y)

# load bilstm model
from utils import LSTM 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load LSTM from checkpoint
model = LSTM.load_from_checkpoint('/nfs_home/nallapar/final/riboclette/riboclette/multimodality/bilstm/saved_models/LSTM DS: Liver [0.3, 20, 0.05, BS 1, D 0.1 E 50 LR 0.0001] F: cbert_ae_codon_ss/epoch=11-step=58848.ckpt', dropout_val=0.1, num_epochs=50, bs=1, lr=1e-4)
model = model.to(device)
# do captum attribution
lga = LayerGradientXActivation(model, model.embedding)

index_val = torch.tensor(96).to(device)
data.x = data.x.to(device)

print(data.x.shape)

data.x = data.x.unsqueeze(0)
data.x = data.x.unsqueeze(0)

attributions, delta = lga.attribute(data.x, target=index_val)

print(attributions.shape, data.x.shape)

# plot attributions as a line plot


