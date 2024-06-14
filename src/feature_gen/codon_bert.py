# library imports
from tokenizer import mytok, get_tokenizer
from transformers import BertForPreTraining
import pandas as pd
import torch
import itertools

def codonidx_to_ntsequence(codonidx_seq):
    codonidx_seq = codonidx_seq[1:-1].split(',')
    codonidx_seq = [int(k) for k in codonidx_seq]
    nt_seq = ''
    for idx in codonidx_seq:
        nt_seq += id_to_codon[idx]

    return nt_seq

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

df_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/LIVER.csv'
# load data
df_full = pd.read_csv(df_path)

print(df_full)

# apply codonidx_to_ntsequence on codon_sequence column to get nt_sequence column
df_full['sequence'] = df_full['codon_sequence'].apply(lambda x: codonidx_to_ntsequence(x))

df_full = df_full[df_full['sequence'].apply(lambda x: len(x)) <= 1022*3]
df_full = df_full[df_full['sequence'].apply(lambda x: len(x)) > 1*3]

model = BertForPreTraining.from_pretrained('/nfs_home/nallapar/final/riboclette/riboclette/multimodality/feature_gen/CodonBERT/')
model = model.cuda()
model.eval()

tokenizer = get_tokenizer()

# get sequences
seqs = df_full['sequence'].tolist()

mytok_seqs = [mytok(seq, 3, 3) for seq in seqs]
tokenized_seqs = [tokenizer.encode(" ".join(x)).ids for x in mytok_seqs]

# get embeddings
codonbert_embeds = []

for i in range(len(tokenized_seqs)):
    input_ids = torch.tensor([tokenized_seqs[i]], dtype=torch.int64).cuda()
    with torch.no_grad():
        print(i, len(tokenized_seqs[i]))
        outputs = model(input_ids, labels=input_ids, output_hidden_states=True)
        _, _, hidden_states = outputs[:3]
        output_embeds = torch.squeeze(hidden_states[-1])[1:-1]
        output_embeds = output_embeds.cpu().numpy()
        codonbert_embeds.append(output_embeds)

df_full['cbert_embeds'] = codonbert_embeds

df_full.to_pickle('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/LIVER_CodonBERT.pkl')

print("Made Embeddings")