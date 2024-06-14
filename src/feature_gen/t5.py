# libraries
import numpy as np
from bio_embeddings.embed import ProtTransT5BFDEmbedder
import pandas as pd 
import itertools

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

codon_to_aa = {
        'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
        'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
        'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
        'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',                
        'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
        'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
        'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
        'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
        'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
        'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
        'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
        'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
        'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
        'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
        'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_',
        'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W',
        'NNG':'R', 'NNC':'T', 'NGT':'S', 'NGA':'R',
        'NNT':'Y', 'NGC':'S'
    }

def get_codon_to_aa(nt_seq):
    aa_seq = ''
    for i in range(0, len(nt_seq), 3):
        codon = nt_seq[i:i+3]
        aa_seq += codon_to_aa[codon]

    aa_seq = aa_seq.replace('_', '')

    return aa_seq

def codonidx_to_ntsequence(codonidx_seq):
    codonidx_seq = codonidx_seq[1:-1].split(',')
    codonidx_seq = [int(k) for k in codonidx_seq]
    nt_seq = ''
    for idx in codonidx_seq:
        nt_seq += id_to_codon[idx]

    return nt_seq

print("Starting")
embedder = ProtTransT5BFDEmbedder()
print("Loaded")

df_path = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/LIVER.csv'
# load data
df_full = pd.read_csv(df_path)

print(df_full)

# apply codonidx_to_ntsequence on codon_sequence column to get nt_sequence column
df_full['sequence'] = df_full['codon_sequence'].apply(lambda x: codonidx_to_ntsequence(x))

df_full = df_full[df_full['sequence'].apply(lambda x: len(x)) <= 1022*3]
df_full = df_full[df_full['sequence'].apply(lambda x: len(x)) > 1*3]

nt_sequences = list(df_full['sequence'])

protein_seqs = []

for i in range(len(nt_sequences)):
    protein_seqs.append(get_codon_to_aa(nt_sequences[i]))

i = 0

embeddings = []
for seq in protein_seqs:
    if len(seq) > 10000:
        big_seq_len = len(seq)
        x = 0
        diff = 5000
        embed_arr = []
        while x < big_seq_len:
            embed_arr.append(np.asarray(embedder.embed(seq[x:min(big_seq_len, x+diff)])))
            x += diff
        append_arr = np.concatenate((embed_arr), axis=0)
        print(append_arr.shape)
        embeddings.append(np.asarray(append_arr))
    else:
        embeddings.append(np.asarray(embedder.embed(seq)))

df_full['protein_embeddings'] = embeddings

df_full.to_pickle('/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/processed/mm/LIVER_t5.pkl')

