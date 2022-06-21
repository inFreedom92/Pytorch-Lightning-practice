# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 18:36:02 2022

@author: swnam
"""



import os
import random



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl

from torchmetrics import Accuracy

pl.utilities.seed.seed_everything(1234)

class LitMNIST(pl.LightningModule):
    def __init__(self, data_dir="", lr=1e-3, batch_size = 1024):
        super().__init__()

        self.data_dir = data_dir
        self.lr = lr
        self.batch_size = batch_size

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            ])
        
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.1),
            nn.Flatten(),
            nn.Linear(32*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes),
        )
        
        

        self.accuracy = Accuracy()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.accuracy(preds, y)
        
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", self.accuracy, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


    def prepare_data(self):
        torchvision.datasets.MNIST(self.data_dir, train=True, download=True)
        torchvision.datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        dataset = torchvision.datasets.MNIST(self.data_dir, train=True, download=False, transform=self.transform)
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(dataset, [55000, 5000])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        test_dataset = torchvision.datasets.MNIST(self.data_dir, train=False, download=False, transform=self.transform)
        return torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size)


model = LitMNIST()


trainer = pl.Trainer(
    gpus=1,
    max_epochs=3,
    progress_bar_refresh_rate=10,
    auto_lr_find=True,
)
trainer.tune(model)


trainer.fit(model)

trainer.test()







