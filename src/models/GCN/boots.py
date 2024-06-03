import pandas as pd 
from scipy.stats import bootstrap, pearsonr
# import mae error
from sklearn.metrics import mean_absolute_error as mae_error
# do bootstrap to get the confidence interval
import numpy as np
from sklearn.utils import resample

# ds = pd.read_csv('predictions/OnlyGNN_DirSeq+.csv')

# perf = list(ds['mae_dsp'])

'''
mimo
'''

# load npy files
pred = np.load('predictions/density_pred.npy', allow_pickle=True)
true = np.load('predictions/density_true.npy', allow_pickle=True)

perf = []
mae = []

for i in range(len(pred)):
    perf.append(pearsonr(pred[i], true[i])[0])
    mae.append(mae_error(pred[i], true[i]))

# configure bootstrap
# perf = mae
n_iterations = 1000
n_size = len(perf)

# run bootstrap
stats = list()
for i in range(n_iterations):
    # prepare train and test sets
    test = resample(perf, n_samples=n_size)
    # calculate statistic
    stat = np.mean(test)
    stats.append(stat)

print(np.mean(stats), '+-' ,1.96 * np.std(stats))

# perf = (perf,)
# bootstrap_ci = bootstrap(perf, np.mean, confidence_level=0.95,
#                          random_state=1, method='percentile')

# #view 95% boostrapped confidence interval
# print(bootstrap_ci.confidence_interval)

''' Perf and MAE
DirSeq+: 63.77529057472295 +- 0.3944343308989013 [63.38085624382405, 64.16972490562185] || 0.037152070414070337 +- 0.0006646153108024218
DirSeq: 64.35131115349375 +- 0.38283321001644377 [63.96847794347731, 63.96847794347731] || 0.037186511972700007 +- 0.0007228965012191325
USeq+: 62.79062943163319 +- 0.3475390875607021 [62.44309034407249, 63.1381685191939] || 0.03754195016670846 +- 0.0007317890615511979
USeq: 63.1760501031053 +- 0.3537684546673195 [62.82228164843798, 63.529818557772614] || 0.03740064835090134 +- 0.0007569268406056821
LSTM: 0.6356173354143573 +- 0.003848151929267712 || 0.03732832161550669 +- 0.0007039766059349285
OnlyGNN - DirSeq+: 0.47448689286155493 +- 0.0043649425828677785 || 0.03969659040236806 +- 0.0008128396266455561
MIMO: 0.4459706685193628 +- 0.008729491691026957 || 0.64931715 +- 0.013802355732768773
'''
