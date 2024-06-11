import os
import torch
from calm import CaLM
import itertools
from tqdm import tqdm

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def codonInt2Seq(codon_int):
    codon_seq_str = ''
    for el in codon_int:
        codon_seq_str += id_to_codon[el]

    return codon_seq_str

# get all the files in folder
inp_folder = '/nfs_home/nallapar/final/DirSeqPlusRW/train/'

# filenames 
files = os.listdir(inp_folder)

# iterate over the files
model = CaLM()

count = 0
for i in tqdm(range(3447, len(files))):
    print(count)
    count += 1
    # load file
    data = torch.load(inp_folder + files[i])
    # get the embeddings
    data.x['codon_seq'] = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)

    codon_seq = codonInt2Seq(data.x['codon_seq'].numpy())
    data.x['calm_embeds'] = model.embed_sequence(codon_seq, average=False)[:, 1:-1, :]

    # save this file
    torch.save(data, '/nfs_home/nallapar/final/DirSeqPlusRW/train/' + files[i])
