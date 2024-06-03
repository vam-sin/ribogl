# libraries
import pandas as pd 
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer
from sklearn.model_selection import train_test_split
import itertools
import os
import lightning as L
from scipy import sparse
from torch.autograd import Variable

id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

def slidingWindowZeroToNan(a, window_size=30):
    '''
    use a sliding window, if all the values in the window are 0, then replace them with nan
    '''
    a = np.asarray(a)
    for i in range(len(a) - window_size):
        if np.all(a[i:i+window_size] == 0.0):
            a[i:i+window_size] = np.nan

    return a

class FileRiboDataset(Dataset):
    def __init__(self, data_folder, dataset_split, shuffle):
        super().__init__()

        self.data_folder = data_folder
        self.dataset_split = dataset_split

        # get all filenames in the folder
        self.files = os.listdir(data_folder + dataset_split)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # shuffle files
        if shuffle:
            np.random.shuffle(self.files)

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # load the file
        data = torch.load(self.data_folder + self.dataset_split + '/' + self.files[idx])
        data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)
        data.y = data.y / torch.nansum(data.y)

        return data.x, data.y
    
class MaskedPearsonLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        
        cos = nn.CosineSimilarity(dim=0, eps=eps)

        cos_val = cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

        return 1 - cos_val
    
class MaskedPearsonCorr(nn.Module):
    def __init__(self):
        super().__init__()
    def __call__(self, y_pred, y_true, mask, eps=1e-6):
        y_pred_mask = torch.masked_select(y_pred, mask)
        y_true_mask = torch.masked_select(y_true, mask)
        cos = nn.CosineSimilarity(dim=0, eps=eps)
        return cos(
            y_pred_mask - y_pred_mask.mean(),
            y_true_mask - y_true_mask.mean(),
        )

class MaskedL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())
    
class MaskedPCCL1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = MaskedL1Loss()
        self.pcc_loss = MaskedPearsonLoss()

    def __call__(self, y_pred, y_true, mask):
        '''
        loss is the sum of the l1 loss and the pearson correlation coefficient loss
        '''

        l1 = self.l1_loss(y_pred, y_true, mask)
        pcc = self.pcc_loss(y_pred, y_true, mask)

        return l1 + pcc, pcc, l1

class LSTM(L.LightningModule):
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        self.bilstm = nn.LSTM(128, 128, num_layers = 4, bidirectional=True)
        self.embedding = nn.Embedding(64, 128)
        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs
        self.perf_list = []
        self.mae_list = []
        self.out_tr = []

    def forward(self, x):
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x = self.embedding(x)
        x = x.squeeze(dim=1)
        x = x.permute(1, 0, 2)

        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        # linear out
        x = self.linear(x)
        x = x.squeeze(dim=1)
        
        # extra for lstm
        out = x.squeeze(dim=1)

        # softmax
        out = self.softmax(out)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y = batch

        y = y.squeeze(dim=0)

        # pass through model
        y_pred = self.forward(x)

        # calculate loss
        lengths = torch.tensor([y.shape[0]]).to(y_pred)
        mask = torch.arange(y_pred.shape[0])[None, :].to(lengths) < lengths[:, None]
        mask = torch.logical_and(mask, torch.logical_not(torch.isnan(y)))

        # squeeze mask
        mask = mask.squeeze(dim=0)

        loss, cos, mae = self.loss(y_pred, y, mask)

        perf = 1 - cos

        return loss, perf, mae

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.num_epochs, eta_min=0)
        return [optimizer], [scheduler]
    
    def training_step(self, batch):

        loss, perf, mae = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae = self._get_loss(batch)

        self.log('val_loss', loss)
        self.log('val_perf', perf)

        return loss
    
    def test_step(self, batch):
        x, y = batch
        loss, perf, mae = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_perf', perf)

        # convert perf to float
        perf = perf.item()
        self.perf_list.append(perf)

        self.mae_list.append(mae.item())

        if len(self.perf_list) == 1284:
            # convert to df
            df = pd.DataFrame({'perf_lstm': self.perf_list, 'mae_lstm': self.mae_list})
            # save to csv
            df.to_csv('lstm.csv')

        return loss
    
def trainLSTM(num_epochs, bs, lr, save_loc, wandb_logger, train_loader, test_loader, dropout_val):
    # Create a PyTorch Lightning trainer with the generation callback
    trainer = L.Trainer(
        default_root_dir=save_loc,
        accelerator="auto",
        devices=1,
        accumulate_grad_batches=bs,
        max_epochs=num_epochs,
        logger=wandb_logger,
        callbacks=[
            L.pytorch.callbacks.ModelCheckpoint(dirpath=save_loc,
                monitor='val_loss',
                save_top_k=2),
            L.pytorch.callbacks.LearningRateMonitor("epoch"),
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=20),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # Check whether pretrained model exists. If yes, load it and skip training
    model = LSTM(dropout_val, num_epochs, bs, lr)
    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False, ckpt_path="best")
    result = {"test": test_result}

    # load model
    # model = LSTM.load_from_checkpoint(save_loc+ '/epoch=7-step=39232.ckpt', dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr)

    # # Test best model on test set
    # test_result = trainer.test(model, dataloaders = test_loader, verbose=False)
    # result = {"test": test_result}

    return model, result
    

        

