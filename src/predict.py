import numpy as np
import pandas as pd
import torch
from torch.utils import data as torch_data
import os
import argparse

from model import Model
from data_retriever import TestDataRetriever
from config import CFG
from utils import get_settings


def main(device, settings):
    models = []
    for i in range(1, CFG.n_fold + 1):
        model = Model()
        model.to(device)
        checkpoint = torch.load(os.path.join(settings['MODEL_CHECKPOINT_DIR'], f'best-model-{i}.pth'))
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        models.append(model)
    
    submission = pd.read_csv(os.path.join(settings['DATA_PATH'], "sample_submission.csv"))

    test_data_retriever = TestDataRetriever(
        settings['TEST_DATA_PATH'],
        submission["BraTS21ID"].values,
        n_frames=CFG.n_frames,
        img_size=CFG.img_size
    )

    test_loader = torch_data.DataLoader(
        test_data_retriever,
        batch_size=4,
        shuffle=False,
        num_workers=8,
    )

    y_pred = []
    ids = []
    for e, batch in enumerate(test_loader):
        print(f"{e}/{len(test_loader)}", end="\r")
        with torch.no_grad():
            tmp_pred = np.zeros((batch["X"].shape[0], ))
            for model in models:
                tmp_res = torch.sigmoid(model(batch["X"].to(device))).cpu().numpy().squeeze()
                tmp_pred += tmp_res
            tmp_pred = tmp_pred/len(models)
            y_pred.extend(tmp_pred)
            ids.extend(batch["id"].numpy().tolist())

    submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": y_pred})
    submission.to_csv(os.path.join(settings['SUBMISSION_DIR'], "submission.csv"), index=False)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--setting-path', default='../settings/SETTINGS.json')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = get_settings(args.setting_path)
    main(device, settings)