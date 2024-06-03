import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# folder
folder_path = '/nfs_home/nallapar/final/riboclette/riboclette/models/xlnet/data/sh/'

test = pd.read_csv(folder_path + 'test_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')
train = pd.read_csv(folder_path + 'train_OnlyLiver_Cov_0.3_NZ_20_PercNan_0.05.csv')

# merge them both
df = pd.concat([train, test], axis=0)

print(df)

cov_mod = list(df['coverage_mod'])

# make a histogram of the coverage
plt.figure(figsize=(5, 5))
plt.hist(cov_mod, bins=100, alpha=0.75, color='#eb4d4b')
plt.xlabel('Coverage', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
plt.title('Coverage Distribution \n in the Mouse Liver Dataset', fontsize=15)
plt.savefig('coverage_dist.svg')