# libraries
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import itertools
import os
import lightning as L
from torch.autograd import Variable

# dictionary for converting codons into one-hot and vice versa
id_to_codon = {idx:''.join(el) for idx, el in enumerate(itertools.product(['A', 'T', 'C', 'G'], repeat=3))}
codon_to_id = {v:k for k,v in id_to_codon.items()}

class FileRiboDataset(Dataset):
    '''
    dataset class which creates a pytorch dataset given the following arguments
    data_folder: the folder where the data is stored
    dataset_split: the split of the dataset (train, test)
    shuffle: whether to shuffle the dataset
    '''
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
        file_name = self.files[idx]
        data = torch.load(self.data_folder + self.dataset_split + '/' + self.files[idx])

        # convert the codon_seq to one-hot
        data.x = torch.tensor([int(k) for k in data.x['codon_seq']], dtype=torch.long)

        # normalize the y values
        data.y = data.y / torch.nansum(data.y)

        return data.x, data.y, file_name
    
class MaskedPearsonLoss(nn.Module):
    '''
    loss function which calculates the pearson correlation coefficient between the predicted and true values
    '''
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
    '''
    metric class for the pearson correlation coefficient
    '''
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
    '''
    loss function which calculates the l1 loss between the predicted and true values
    '''
    def __init__(self):
        super().__init__()

    def __call__(self, y_pred, y_true, mask):
        y_pred_mask = torch.masked_select(y_pred, mask).float()
        y_true_mask = torch.masked_select(y_true, mask).float()

        loss = nn.functional.l1_loss(y_pred_mask, y_true_mask, reduction="none")
        return torch.sqrt(loss.mean())
    
class MaskedPCCL1Loss(nn.Module):
    '''
    loss function that incorporates both the pearson loss and the l1 loss together
    '''
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
    '''
    LSTM model for learning ribosome density values. this can be initialized with the following arguments:
    dropout_val: dropout value for the LSTM
    num_epochs: number of epochs to train the model
    bs: batch size
    lr: learning rate
    '''
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        # initializes a bidirectional long short term memory model with 4 layers of 128 nodes each
        self.bilstm = nn.LSTM(256, 128, num_layers = 4, bidirectional=True, dropout=dropout_val)
        # learning embedding layer of 256 values
        self.embedding = nn.Embedding(64, 256)

        # linear layer for final output
        self.linear = nn.Linear(256, 1)
        
        # activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        # loss functions and performance metrics
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        # hyperparameters
        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

    def forward(self, x):
        # bilstm initial hidden and cell states
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # codon embeddings
        x = self.embedding(x)
        x = x.permute(1, 0, 2)

        # pass through bilstm
        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        # linear out
        x = self.linear(x)
        x = x.squeeze(dim=1)
        
        # dimension matching
        out = x.squeeze(dim=1)

        # output softmax
        out = self.softmax(out)

        return out
    
    def _get_loss(self, batch):
        # get features and labels
        x, y, file_name = batch

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

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_loss', loss)
        self.log('val_perf', perf)

        return loss
    
    def test_step(self, batch):
        x, y, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_perf', perf)

        return loss

class LSTMPred(L.LightningModule):
    '''
    additional version of the LSTM model made to predict ribosome density values
    '''
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        self.bilstm = nn.LSTM(256, 128, num_layers = 4, bidirectional=True, dropout=dropout_val)
        self.embedding = nn.Embedding(64, 256)
        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

    def forward(self, x):
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x = self.embedding(x)
        x = torch.unsqueeze(x, dim=0)
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
        x, y, file_name = batch

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

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_loss', loss)
        self.log('val_perf', perf)

        return loss
    
    def test_step(self, batch):
        x, y, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_perf', perf)

        return loss

class LSTM_Captum(L.LightningModule):
    '''
    additional captum based lstm model to make the attributions
    '''
    def __init__(self, dropout_val, num_epochs, bs, lr):
        super().__init__()

        self.bilstm = nn.LSTM(256, 128, num_layers = 4, bidirectional=True, dropout=dropout_val)
        self.embedding = nn.Embedding(64, 256)
        self.linear = nn.Linear(256, 1)
        
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        
        self.loss = MaskedPCCL1Loss()
        self.perf = MaskedPearsonCorr()

        self.lr = lr
        self.bs = bs
        self.num_epochs = num_epochs

    def forward(self, x, index_attr):
        # bilstm final layer
        h_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)
        c_0 = Variable(torch.zeros(8, 1, 128).cuda()) # (1, bs, hidden)

        # switch dims for lstm
        x = self.embedding(x)
        x = torch.unsqueeze(x, 0)
        x = x.permute(1, 0, 2)

        x, (fin_h, fin_c) = self.bilstm(x, (h_0, c_0))

        # linear out
        x = self.linear(x)
        x = x.squeeze(dim=1)
        
        # extra for lstm
        out = x.squeeze(dim=1)

        # softmax
        out = self.softmax(out)

        return out[index_attr]
    
    def _get_loss(self, batch):
        # get features and labels
        x, y, file_name = batch

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

        return loss, perf, mae, y_pred

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)
        monitor = 'val_loss'
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
    
    def training_step(self, batch):

        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('train_loss', loss, batch_size=self.bs)
        self.log('train_perf', perf, batch_size=self.bs)

        return loss
    
    def validation_step(self, batch):
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('val_loss', loss)
        self.log('val_perf', perf)

        return loss
    
    def test_step(self, batch):
        x, y, file_name = batch
        loss, perf, mae, y_pred = self._get_loss(batch)

        self.log('test_loss', loss)
        self.log('test_mae', mae)
        self.log('test_perf', perf)

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
            L.pytorch.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ],
    )
    trainer.logger._log_graph = False  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need

    # create the model
    model = LSTM(dropout_val, num_epochs, bs, lr)
    # fit trainer
    trainer.fit(model, train_dataloaders = train_loader, val_dataloaders = test_loader)
    # Test best model on test set
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False, ckpt_path="best")
    result = {"test": test_result}

    # load model
    # model = LSTM.load_from_checkpoint(save_loc+ '/epoch=11-step=67092.ckpt', dropout_val=dropout_val, num_epochs=num_epochs, bs=bs, lr=lr)

    # # Test best model on test set
    # test_result = trainer.test(model, dataloaders = test_loader, verbose=False)
    # result = {"test": test_result}

    return model, result
    

        

