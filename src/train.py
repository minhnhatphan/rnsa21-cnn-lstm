import time
import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
from torch.nn import functional as F
import os
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import argparse

from model import Model
from config import CFG
from utils import LossMeter, AccMeter, seed_everything, get_settings
from trainer import Trainer
from data_retriever import DataRetriever
    
def main(device, settings):
    df = pd.read_csv(os.path.join(settings['DATA_PATH'], "train_labels.csv"))
    train_transform = A.Compose([
                                A.HorizontalFlip(p=0.5),
                                A.ShiftScaleRotate(
                                    shift_limit=0.0625, 
                                    scale_limit=0.1, 
                                    rotate_limit=10, 
                                    p=0.5
                                ),
                                A.RandomBrightnessContrast(p=0.5),
                            ])
    skf = StratifiedKFold(n_splits=CFG.n_fold)
    t = df['MGMT_value']

    start_time = time.time()
    losses = []
    scores = []
    for fold, (train_index, val_index) in enumerate(skf.split(np.zeros(len(t)), t), 1):
        print('-'*30)
        print(f"Fold {fold}")
        
        train_df = df.loc[train_index]
        val_df = df.loc[val_index]

        train_retriever = DataRetriever(
            settings['TRAIN_DATA_PATH'],
            train_df["BraTS21ID"].values, 
            train_df["MGMT_value"].values,
            n_frames=CFG.n_frames,
            img_size=CFG.img_size,
            transform=train_transform
        )
        val_retriever = DataRetriever(
            settings['TRAIN_DATA_PATH'],
            val_df["BraTS21ID"].values, 
            val_df["MGMT_value"].values,
            n_frames=CFG.n_frames,
            img_size=CFG.img_size
        )

        train_loader = torch_data.DataLoader(
            train_retriever,
            batch_size=8,
            shuffle=True,
            num_workers=8,
        )
        valid_loader = torch_data.DataLoader(
            val_retriever, 
            batch_size=8,
            shuffle=False,
            num_workers=8,
        )
        
        model = Model(cnn_path=settings['PRETRAINED_CHECKPOINT_PATH'])
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        criterion = F.binary_cross_entropy_with_logits
        trainer = Trainer(
            model, 
            device, 
            optimizer, 
            criterion, 
            LossMeter, 
            AccMeter
        )
        
        loss, score = trainer.fit(
            CFG.n_epochs, 
            train_loader, 
            valid_loader, 
            os.path.join(settings["MODEL_CHECKPOINT_DIR"], f"best-model-{fold}.pth"), 
            100,
        )
        losses.append(loss)
        scores.append(score)
        trainer.plot_loss()
        trainer.plot_score()
        
    elapsed_time = time.time() - start_time
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(elapsed_time // 60, elapsed_time % 60))
    print('Avg loss {}'.format(np.mean(losses)))
    print('Avg score {}'.format(np.mean(scores)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting-path', default='../settings/SETTINGS.json')
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    settings = get_settings(args.setting_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    main(device, settings)