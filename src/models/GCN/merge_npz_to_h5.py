import numpy as np
import h5py 
import os

filename = '/net/lts2gdk0/mnt/scratch/lts2/nallapar/rb-prof/data/Jan_2024/Lina/full_Test_LIG_RiboGL.h5'

input_folder = '/nfs_home/nallapar/final/riboclette/riboclette/ribogl_int'

len_files = len(os.listdir(input_folder))

# h5py dataset
out_ds = h5py.File(filename, 'w')

# make datasets in out_ds
node_attr_ds = out_ds.create_dataset(
    'node_attr',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

edge_attr_ds = out_ds.create_dataset(
    'edge_attr',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

y_pred_ds = out_ds.create_dataset(
    'y_pred',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

edge_index_ds = out_ds.create_dataset(
    'edge_index',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

transcript_ds = out_ds.create_dataset(
    'transcript',
    (len_files,),
    dtype=h5py.special_dtype(vlen=str)
)

y_true_ds = out_ds.create_dataset(
    'y_true',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

x_input_ds = out_ds.create_dataset(
    'x_input',
    (len_files,),
    dtype=h5py.special_dtype(vlen=np.dtype('float64'))
)

count = 0
# loop through files and add the data to the h5 file
for i, file in enumerate(os.listdir(input_folder)):
    print(count, len_files)
    count += 1
    with np.load(os.path.join(input_folder, file), allow_pickle=True) as data:
        data = data['arr_0']
        data = data.item()
        x_input_ds[i] = data['x_input']
        y_true_ds[i] = data['y_true']
        transcript_ds[i] = data['transcript']
        # flatten edge_index
        edge_index_ds[i] = data['edge_index'].flatten()
        y_pred_ds[i] = data['y_pred']
        edge_attr_ds[i] = data['edge_attr_ds']
        node_attr_ds[i] = data['node_attr_ds']

