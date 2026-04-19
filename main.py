# -*- coding: utf-8 -*-
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '.'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import torch
import os
from pathlib import Path
from utils.loss import define_loss
from sklearn.model_selection import KFold
from utils.utils import seed_torch,pretty_print_args
from Datasets_builder.Graph_Data_builder import get_Graph_data,infer_microbe_dim,get_Moe_data
from torch.utils.data import Subset
from models.main_model import HyMMSurv
from models.Engine import Engine
import pandas as pd


def main(args):
    if torch.cuda.is_available():
        device_ids = [int(x) for x in args.device.split(',')]
        torch.cuda.set_device(device_ids[0])
        main_device = torch.device(f"cuda:{device_ids[0]}")
        print(f"Using GPUs: {device_ids}")
    else:
        main_device = torch.device("cpu")
        print("Using CPU.")
  
    seed_torch(main_device)

    file_name = Path(args.data_path).stem
    args.num_patches = int(file_name.split("_")[-1]) * 2 

    pretty_print_args(args)

    save_folder = os.path.join(args.results_base, str(file_name)+"_Kinit_"+str(args.k_init) + "_epoch_" + str(args.epochs))
    os.makedirs(save_folder, exist_ok=True)
    # Loading data
    if args.model in ["HyMMSurv"]:
        dataset = get_Moe_data(args.data_path)
    else:
        dataset = get_Graph_data(args.data_path)

    microbe_dim = infer_microbe_dim(dataset)

    fold_results = []  
    k_fold5 = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold_idx, (train_idx, test_idx) in enumerate(k_fold5.split(dataset)):
        train_subset = Subset(dataset, train_idx)
        test_subset = Subset(dataset, test_idx)
        print(f"Fold {fold_idx}: train data = {len(train_subset)}, valia data = {len(test_subset)}")
       
        if args.model == "HyMMSurv":
            model = HyMMSurv(device=main_device,microbe_dim=microbe_dim,k_init=args.k_init)
        model = model.to(main_device)

  
        criterion = define_loss("nll_surv")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        save_dir = os.path.join(save_folder, "fold_" + str(fold_idx + 1))
        os.makedirs(save_dir, exist_ok=True)

        engine = Engine(save_folder, fold_idx, args)
        best_score, best_epoch, risk_csv = engine.run(
            model=model,
            train_dataset=train_subset,
            val_dataset=test_subset,
            criterion=criterion,
            optimizer=optimizer,
            fold_idx=fold_idx + 1
        )

        fold_risk_csv_path = os.path.join(save_dir, f"fold_{fold_idx + 1}_risk.csv")
        risk_csv.to_csv(fold_risk_csv_path, index=False)
        
        KM_path_path = os.path.join(save_dir, f"fold_{fold_idx + 1}_km_curve.png")
 
        fold_results.append({
            "fold": fold_idx + 1,
            "best_c_index": best_score,
            "best_epoch": best_epoch
        })

    summary_df = pd.DataFrame(fold_results)
    summary_csv_path = os.path.join(save_folder, "5fold_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

  
    return 0
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SurMoE model with microbe + pathology data")
    parser.add_argument('--device', type=str, default='0',
                        help='comma-separated GPU device ids, e.g. "0,1,2"')
    parser.add_argument("--data_path", type=str, required=True,
                        help="data path")
    parser.add_argument("--results_base", type=str,
                        required=True,
                        help="Base results folder")
    parser.add_argument(
        "--model",
        type=str,
        choices=["HyMMSurv"],
        default="HyMMSurv",
        help="choose which model to run"
    )
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    # parser.add_argument("--num_patches", type=int, default=4000,
    #                     help="Number of num_patches for each WSI")
    parser.add_argument("--batchsize", type=int, default=1,
                        help="batchsize")
    parser.add_argument("--k_init", type=int, default=8,
                        help="跨模态选择top k 注意力的patch块")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                        help="Weight decay for optimizer")
    args = parser.parse_args()
    main(args)