# -*- coding: utf-8 -*-

import sys
import os
from sksurv.metrics import concordance_index_censored
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../HyMMSurv'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
import pickle
import os
import torch
import h5py
from functools import partial
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from utils.utils import *
from Datasets_builder.Graph_Data_builder import collate_multi_HyMMSurv


class Engine(object):
    def __init__(self, save_folder, fold, args):
        save_dir = os.path.join(save_folder, "fold_" + str(fold + 1))
        os.makedirs(save_dir, exist_ok=True)
        self.results_dir = save_dir
        self.save_folder = save_folder
        self.fold = fold
        self.filename_best = None
        self.epoch = 0
        self.best_score = 0
        self.best_epoch = 0
        self.args = args
        self.batchsize = args.batchsize
        self.batchsize = 1 
        self.collate_fn = collate_multi_HyMMSurv 
        self.val_samples_attention = []  

    def run(self, model, train_dataset, val_dataset, criterion, optimizer, fold_idx):
        train_loss_total = []
        train_loss_sur = []
        train_constrct = []
        val_loss_class = []

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn
        )

        for epoch in range(self.args.epochs):
            self.epoch = epoch
            loss_meter, train_c_index = self.train_epoch(model, train_dataloader, criterion, optimizer)

            risk_csv, val_c_index, val_loss = self.validate(model, val_dataloader, criterion)
            print('[Epoch {}] Train --- sur_loss: {:.4f}, ''c_index: {:.4f}'.format(self.epoch, loss_meter["loss_sur"],
                                                                                    train_c_index))
            print('[Epoch {}] Valia --- sur_loss: {:.4f}, c_index: {:.4f}'.format(self.epoch, val_loss, val_c_index))
            print()
            
            train_loss_total.append(loss_meter["loss_total"])
            train_loss_sur.append(loss_meter["loss_sur"])
            train_constrct.append(loss_meter["unsup_loss"] + loss_meter["sup_loss"])
            val_loss_class.append(val_loss)

            # save best
            is_best = val_c_index > self.best_score
            if is_best:
                self.best_score = val_c_index
                self.best_epoch = self.epoch
                self.save_checkpoint({
                    'epoch': epoch,
                    'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'best_score': self.best_score
                })
                risks = risk_csv
                if self.args.model == "M2Surv" or "M2Surv_3model" or "M2Surv_genesPath":
                    
                    save_atten_path = os.path.join(self.results_dir, "all_val_attn.pkl")
                    with open(save_atten_path, "wb") as f:
                        pickle.dump(self.val_samples_attention, f)
            self.val_samples_attention = []

        print("\n" + "=" * 40)
        print(f"Validation Best Results at Epoch {self.best_epoch}")
        print("best c-index={:.4f}".format(self.best_score))
        print("=" * 40 + "\n")

        return self.best_score, self.best_epoch, risks

    def train_epoch(self, model, dataloader, criterion, optimizer):
        model.train()
 
        N = len(dataloader.dataset)

        if hasattr(model, 'module'):
            main_device = next(model.module.parameters()).device
        else:
            main_device = next(model.parameters()).device

        loss_meter = {
            "loss_total": 0,
            "loss_sur": 0,
            "unsup_loss": 0,
            "sup_loss": 0
        }
        all_risk_scores = np.zeros(N)
        all_censorships = np.zeros(N)
        all_event_times = np.zeros(N)

        start_idx = 0 
        for batch_idx, batch in enumerate(dataloader):
            B = 1
            end_idx = start_idx + B
            batch_index = np.arange(start_idx, end_idx)
            optimizer.zero_grad()
            label = batch["label"].to(main_device)
            risk_label = batch["risk_label"].to(main_device)

            hazards, out, S = model(**batch)
            loss_sur = criterion(hazards=hazards, S=S, Y=label, c=risk_label)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()

            loss = loss_sur

            if torch.isnan(loss):
                print("Found invalid loss at batch", batch_idx)
                print("sample_name:", batch["sample_name"])
                # print("batch_num:", len(batch))
                if torch.isnan(loss_sur):
                    print('loss_sur loss is NaN')
                raise ValueError
            loss_meter["loss_sur"] += loss_sur.item()
            loss_meter["loss_total"] += loss.item()
            # loss_meter["unsup_loss"] += out["unsup_loss"].item()
            # loss_meter["sup_loss"] += out["sup_loss"].item()

            all_risk_scores[batch_index] = risk
            all_censorships[batch_index] = batch["risk_label"]
            all_event_times[batch_index] = batch["task_time"]

            start_idx = end_idx 

            loss.backward()
            optimizer.step()

        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                             tied_tol=1e-08)[0]
        for key in loss_meter:
            loss_meter[key] /= N

        return loss_meter, c_index

    def validate(self, model, dataloader, criterion):
        model.eval()
        if hasattr(model, 'module'):
            device = next(model.module.parameters()).device
        else:
            device = next(model.parameters()).device

        
        N = len(dataloader.dataset)
        start_idx = 0  

        val_loss = 0.0
        all_risk_scores = np.zeros(N)
        all_censorships = np.zeros(N)
        all_event_times = np.zeros(N)
        risk_list = []
        sample_name_list = []
        task_time_list = []
        c_list = []
        event_list = []


        for batch_idx, batch in enumerate(dataloader):
            B = 1
            end_idx = start_idx + B
            batch_index = np.arange(start_idx, end_idx)
            sample_name_list.extend(batch["sample_name"])
            event_list.extend(batch["label"].numpy().tolist())
            c_list.extend(batch["risk_label"].numpy().tolist())
            task_time_list.extend(batch["task_time"].numpy().tolist())

            label = batch["label"].to(device)
            risk_label = batch["risk_label"].to(device)
            task_time = batch["task_time"].to(device)
            with torch.no_grad():
                hazards, out, S = model(**batch)
                if self.args.model == "HyMMSurv":
                    sample_dict = {
                        "sample_name": batch["sample_name"][0],
                        "attn_ff_20x": out["attn_ff_20x"].detach().cpu().numpy(),
                        "attn_ffpe_10x": out["attn_ffpe_10x"].detach().cpu().numpy(),
                        "attn_path": out["attn_path"].detach().cpu().numpy(),
                        "attn_microb": out["attn_microb"].detach().cpu().numpy(),
                        "coords_20x": batch["coords_20x"],
                        "coords_10x": batch["coords_10x"]
                    }
                    self.val_samples_attention.append(sample_dict)
            loss = criterion(hazards=hazards, S=S, Y=label, c=risk_label)
            risk = -torch.sum(S, dim=1).cpu().numpy()

            all_risk_scores[batch_index] = risk
            all_censorships[batch_index] = batch["risk_label"]
            all_event_times[batch_index] = batch["task_time"]
            val_loss += loss.item()

            risk_list.extend(risk.tolist())
            start_idx = end_idx  

        risk_csv = pd.DataFrame({'sample_name': sample_name_list,
                                 'risk': risk_list,
                                 'even': event_list,
                                 'time': task_time_list,
                                 'risk_cohort': c_list})
        val_loss /= len(dataloader)
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores,
                                             tied_tol=1e-08)[0]
        return risk_csv, c_index, val_loss

    def save_checkpoint(self, state):
        if self.filename_best is not None and os.path.exists(self.filename_best):
            os.remove(self.filename_best)

        self.filename_best = os.path.join(self.results_dir, f'model_best_{state["best_score"]:.4f}_epoch_{state["epoch"]}.pth')
        torch.save(state, self.filename_best)
        print()
        print(f"Saved best model to： {self.filename_best}")
