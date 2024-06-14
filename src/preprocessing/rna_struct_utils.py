# library imports
import pandas as pd
import numpy as np
import RNA
import itertools
from scipy import sparse

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def codonidx_to_ntsequence(codonidx_seq):
    codonidx_seq = codonidx_seq[1:-1].split(',')
    codonidx_seq = [int(k) for k in codonidx_seq]
    nt_seq = ''
    for idx in codonidx_seq:
        nt_seq += id_to_codon[idx]

    return nt_seq

def mergeNTGraphToCodonGraph(nt_adj_mat):
    codon_adj_mat = np.zeros((nt_adj_mat.shape[0]//3, nt_adj_mat.shape[1]//3))
    for i in range(nt_adj_mat.shape[0]//3):
        for j in range(nt_adj_mat.shape[1]//3):
            codon_adj_mat[i][j] = np.sum(nt_adj_mat[i*3:i*3+3, j*3:j*3+3])

    # make principal diagonal 0
    for i in range(codon_adj_mat.shape[0]):
        codon_adj_mat[i][i] = 0
    # binarize the matrix (if val > 0, val = 1)
    codon_adj_mat[codon_adj_mat > 0] = 1

    codon_adj_mat = np.array(codon_adj_mat, dtype=np.int32)

    return codon_adj_mat

def getRNASS(nt_seq):
    ss, mf = RNA.fold(nt_seq)

    len_ss = len(ss)
    # make adj matrix
    adj = np.zeros((len_ss, len_ss))

    # every nt connected to the one right after it
    for j in range(len_ss-1):
        adj[j][j+1] = 1.0
        adj[j+1][j] = 1.0

    # the loops
    stack = []
    for j in range(len_ss):
        if ss[j] == '(':
            stack.append(j)
        elif ss[j] == ')':
            conn_1 = j 
            conn_2 = stack.pop()
            adj[conn_1][conn_2] = 1.0
            adj[conn_2][conn_1] = 1.0
        else:
            pass 

    adj = np.asarray(adj)

    return adj
        



