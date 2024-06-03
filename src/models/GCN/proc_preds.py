import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
sns.set(style="white")

# scatter plot of lstm and dirseq+ performances

dirseqplus = pd.read_csv('predictions/DirSeq+.csv')

lstm = pd.read_csv('predictions/lstm.csv')

# # get the performances
dirseqplus_perf = list(dirseqplus['perf_dsp'])
lstm_perf = list(lstm['perf_lstm'])

df = pd.DataFrame({'dirseqplus_perf': dirseqplus_perf, 'lstm_perf': lstm_perf})
# print(np.mean([dirseqplus_perf[i] - lstm_perf[i] for i in range(len(dirseqplus_perf))]))

# make a fancy scatter plot
plt.figure(figsize=(30, 30))
b = sns.jointplot(data = df, x='dirseqplus_perf', y='lstm_perf', alpha=0.75, color='#eb4d4b')
# # scale the axes
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('DirSeq+ Performance', fontsize=20)
plt.ylabel('LSTM Performance', fontsize=20)

# add title
plt.subplots_adjust(top=0.85)
b.fig.suptitle('Scatter plot with the DirSeq+ \n and LSTM performances', fontsize=20)

# # add a diagonal line
plt.plot([0, 1], [0, 1], transform=plt.gca().transAxes, ls="--", c=".3")

# number of sequences below and above the diagonal
below = sum([1 for i in range(len(dirseqplus_perf)) if dirseqplus_perf[i] > lstm_perf[i]])
above = sum([1 for i in range(len(dirseqplus_perf)) if dirseqplus_perf[i] < lstm_perf[i]])

print('Number of sequences below the diagonal:', below)
print('Number of sequences above the diagonal:', above)
print('Pearson r: ', pearsonr(dirseqplus_perf, lstm_perf))

# save the plot
plt.savefig('scatter_plot.svg')

