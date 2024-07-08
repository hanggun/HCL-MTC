# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 11:46:56 2020

@author: Administrator
"""
import torch
from data_modules.vocab import Vocab
import os
from helper.configure import Configure
from models.model import HiAGM
from data_modules.data_loader import data_loaders
from data_modules.dataset import ClassificationDataset
from data_modules.collator import Collator
from torch.utils.data import DataLoader
import tqdm
import numpy as np
from train_modules.evaluation_metrics import evaluate
from train_modules.criterions import ClassificationLoss
from train_modules. trainer import Trainer
from helper.utils import load_checkpoint, save_checkpoint
import warnings
warnings.filterwarnings('ignore')
def set_optimizer(config, model):
    """
    :param config: helper.configure, Configure Object
    :param model: computational graph
    :return: torch.optim
    """
    params = model.optimize_params_dict()
    if config.train.optimizer.type == 'Adam':
        return torch.optim.Adam(lr=config.train.optimizer.learning_rate,
                                params=params)
    else:
        raise TypeError("Recommend the Adam optimizer")
        


config = Configure(config_json_file='config/gcn-wos.json')
corpus_vocab = Vocab(config,
                    min_freq=5,
                    max_size=50000)
vocab = corpus_vocab
hiagm = HiAGM(config, corpus_vocab, model_type=config.model.type, model_mode='TRAIN')
hiagm.to(config.train.device_setting.device)

optimize = set_optimizer(config, hiagm)

model_checkpoint = config.train.checkpoint.dir
model_name = config.model.type
best_epoch_model_file = os.path.join(model_checkpoint, 'best_macro_' + model_name)
# best_epoch_model_file = os.path.join(model_checkpoint, 'HiAGM-TP_epoch_51')
checkpoint_model = torch.load(best_epoch_model_file, map_location='cuda:2')

hiagm.load_state_dict(checkpoint_model['state_dict'])
optimize.load_state_dict(checkpoint_model['optimizer'])
model = hiagm
model.eval()

predict_probs = []
target_labels = []
total_loss = 0.0

on_memory = False
collate_fn = Collator(config, vocab)
test_dataset = ClassificationDataset(config, vocab, stage='TEST', on_memory=on_memory, corpus_lines=None)
test_loader = DataLoader(test_dataset,
                              batch_size=config.eval.batch_size,
                              shuffle=True,
                              num_workers=config.train.device_setting.num_workers,
                              collate_fn=collate_fn,
                              pin_memory=True)

criterion = ClassificationLoss(os.path.join(config.data.data_dir, config.data.hierarchy),
                               corpus_vocab.v2i['label'],
                               recursive_penalty=config.train.loss.recursive_regularization.penalty,
                               recursive_constraint=config.train.loss.recursive_regularization.flag)

trainer = Trainer(model=hiagm,
                      criterion=criterion,
                      optimizer=optimize,
                      vocab=corpus_vocab,
                      config=config)

if os.path.isfile(best_epoch_model_file):
    load_checkpoint(best_epoch_model_file, model=hiagm,
                    config=config,
                    optimizer=optimize)
    trainer.eval(test_loader, 0, 'TEST')