import h5py
import torch
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

dirseq_preds = h5py.File('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqNoTransform/test_preds_normSoftmax.h5', 'r')['input_x_gradient']
dirseqplus_preds = h5py.File('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqPlusNoTransform/test_preds_normSoftmax.h5', 'r')['input_x_gradient']

true_y = h5py.File('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/DirSeqNoTransform/test_truth_normSoftmax.h5', 'r')['input_x_gradient']

# function to check if there is a 3d edge close to the node
def check_3d_edge(node_id, sample_id):
    file_name = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/USeqPlusNoTransform/test/sample_' + str(sample_id) + '.pt'

    data = torch.load(file_name)

    ei = data.edge_index

    # check if there's a 3d edge close to the node
    for j in range(ei.shape[1]):
        if (ei[0][j] == node_id or ei[1][j] == node_id) and (np.abs(ei[0][j] - ei[1][j]) != 1):
            return True
        
    return False

# open a file to write the results
# file = open('samples_peaks_diff.txt', 'w')

class MaskedPearsonCorr(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = torch.nn.CosineSimilarity(dim=0, eps=eps)
        return cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )
    
perf = MaskedPearsonCorr()

print(dirseq_preds.shape)

dirseqplus_diffs = []
dirseq_diffs = []

for i in range(dirseq_preds.shape[0]):
    # for each of the samples, first order the nodes in truth_y based on their absolute values
    # then get the top 10 nodes from the truth_y
    # check if dirseqplus orders them better than dirseq
    # if yes, then check if there's a 3d edge close to the node
    # if yes, then write the sample number to the file
    # get pearsonr
    # print(pearsonr(dirseq_preds[i].flatten(), dirseqplus_preds[i].flatten()))
    lengths = torch.tensor([true_y[i].shape[0]])
    mask = torch.arange(dirseqplus_preds[i].shape[0])[None, :].to(lengths) < lengths[:, None]
    mask = torch.logical_and(mask, torch.logical_not(torch.isnan(torch.tensor(true_y[i]))))
    pcc_dsp = perf(torch.tensor(dirseqplus_preds[i].flatten()), torch.tensor(true_y[i].flatten()), mask)
    pcc_ds = perf(torch.tensor(dirseq_preds[i].flatten()), torch.tensor(true_y[i].flatten()), mask)

    print(i, pcc_dsp, pcc_ds)

    # get the top 10 nodes from the truth_y
    true_y_sample_vals = true_y[i]
    true_y_sample_vals = true_y_sample_vals.flatten()
    # convert nans to 0
    true_y_sample_vals = np.nan_to_num(true_y_sample_vals)
    # get index of the biggest 10 values
    true_y_sample = np.argsort(true_y_sample_vals)[::-1][:10]

    # print these values 

    # get the ranking of the top 10 nodes in truth_y in dirseqplus and dirseq
    dirseqplus_sample = dirseqplus_preds[i]
    dirseqplus_sample = dirseqplus_sample.flatten()
    dirseqplus_sample = np.argsort(dirseqplus_sample)[::-1]
    # index of the top 10 nodes in dirseqplus
    dirseqplus_sample = np.where(np.isin(dirseqplus_sample, true_y_sample))[0]

    # do the same for dirseq
    dirseq_sample = dirseq_preds[i]
    dirseq_sample = dirseq_sample.flatten()
    dirseq_sample = np.argsort(dirseq_sample)[::-1]
    # index of the top 10 nodes in dirseq
    dirseq_sample = np.where(np.isin(dirseq_sample, true_y_sample))[0]

    # if pcc_dsp > pcc_ds:
    #     for j in range(10):
    #         # check if dirseqplus orders them better than dirseq
    #         if dirseq_sample[j] - dirseqplus_sample[j] > 10:
    #             # if yes, then check if there's a 3d edge close to the node
    #             if check_3d_edge(true_y_sample[j], i):
    #                 # if yes, then write the sample number to the file, along with the node id
    #                 file.write('sample_' + str(i) + ' node_id: ' + str(true_y_sample[j]) + '\n')

    for j in range(10):
        dirseqplus_diffs.append(dirseqplus_sample[j] - j)
        dirseq_diffs.append(dirseq_sample[j] - j)

# make a scatter plot of the differences
# scatter log scale
fig = plt.figure()
ax = plt.gca()
ax.scatter(dirseqplus_diffs, dirseq_diffs, color='#eb4d4b', alpha=0.75)
ax.set_yscale('log')
ax.set_xscale('log')
plt.xlabel('DirSeqPlus - Truth')
plt.ylabel('DirSeq - Truth')
plt.title('Scatter plot of differences in rankings')
plt.savefig('scatter_plot_peakdiffs.png')
            
# file.close()