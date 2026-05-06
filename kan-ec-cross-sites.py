import torch.nn.functional as F
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

import pandas as pd

feature_dict = {
    'temperature': 'TA_F',
    'VPD': 'VPD_F',
    'soil moisture': 'SWC_F_MDS_1',
    'precipitation': 'P_F',
    'soil temperature': 'TS_F_MDS_1',
    'shortwave radiation': 'SW_IN_F',
}

n_run = 2
n_epochs = 10
n_steps = 10

import os
import json

directory_path = './data/site_data'

all_entries = os.listdir(directory_path)

site_files = []
for entry in all_entries:
    root, extension = os.path.splitext(entry)
    full_path = os.path.join(directory_path, entry)
    if extension == ".csv" and os.path.join(directory_path, entry):
        site_files.append(full_path)

from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
input_dim = len(feature_dict)

features = list(feature_dict.values())
target_col = 'RECO_NT_VUT_REF'

site_datasets = {}

for site_file in site_files:
    print(site_file)
    site_name = site_file.split('\\')[-1].split('_')[0]
    df = pd.read_csv(site_file)

    # Drop rows with NaNs in features or target
    df_model = df.dropna(subset=features + [target_col, 'year']).copy()
    df_features, df_labels = df_model[features], df_model[target_col].values.reshape(-1, 1)

    train_input, test_input, train_label, test_label = train_test_split(df_features, df_labels, test_size=0.2, random_state=42)

    train_input = train_input.values
    train_label = np.log(train_label)

    test_input = test_input.values
    test_label = np.log(test_label)

    dataset = {
        "train_input": torch.DoubleTensor(train_input).to(device),
        "train_label": torch.DoubleTensor(train_label).to(device),
        "test_input": torch.DoubleTensor(test_input).to(device),
        "test_label": torch.DoubleTensor(test_label).to(device)
    }

    site_datasets[site_name] = dataset

cross_site_r2s, cross_site_rho2s = {}, {}
for train_site_name, train_dataset in site_datasets.items():
    print(train_site_name)
    print(train_dataset['train_input'].shape, train_dataset['train_label'].shape)

    cross_site_r2s[train_site_name] = {}
    cross_site_rho2s[train_site_name] = {}

    # KAN initialization
    # model = KAN(width=[input_dim, 2 * input_dim + 1, 1], grid=3, k=3, noise_scale=0.1, seed=1, device=device)
    model = KAN(width=[input_dim, 2 * input_dim + 1, 1], grid=3, k=3, noise_scale=0.3, seed=42, device=device)

    # train the model
    model.train()

    r2_train_best, rho2_train_best = -100, -100

    for e in range(n_epochs):
        print(f"Training Epoch {e+1} Starts")
        model.fit(train_dataset, opt="LBFGS", steps=n_steps, lamb=0.005, lamb_entropy=1.0, lr=1)
        # print(f"KAN visualization for test year {test_year}")

        model.eval()
        with torch.no_grad():
            train_pred = model(train_dataset['train_input']).cpu().numpy().ravel()
            train_label = train_dataset['train_label'].cpu().numpy().ravel()

            r2_train = r2_score(np.exp(train_label), np.exp(train_pred))
            rho2_train = (np.corrcoef(np.exp(train_label), np.exp(train_pred))[0,1])**2

        print(f"Evaluation for Epoch {e+1}. Train R² {r2_train:.3f}, Train Rho² {rho2_train:.3f}")

        if r2_train > r2_train_best:
            r2_train_best = r2_train
            rho2_train_best = rho2_train

            model.saveckpt(f"./model/run_{n_run}/kan_{train_site_name}_complete_{n_epochs * n_steps}_model")

    print(f"Best Training Metrics for {train_site_name}. Train R² {r2_train_best:.3f}, Train Rho² {rho2_train_best:.3f}")

    print(f"Cross-site Evaluation for {train_site_name}.")

    model = KAN.loadckpt(f"./model/run_{n_run}/kan_{train_site_name}_complete_{n_epochs * n_steps}_model").to(device)

    for test_site_name, test_dataset in site_datasets.items():
        model.eval()
        with torch.no_grad():
            test_pred = model(test_dataset['test_input']).cpu().numpy().ravel()
            test_label = test_dataset['test_label'].cpu().numpy().ravel()

            r2_test = r2_score(np.exp(test_label), np.exp(test_pred))
            rho2_test = (np.corrcoef(np.exp(test_label), np.exp(test_pred))[0, 1]) ** 2

            cross_site_r2s[train_site_name][test_site_name] = r2_test
            cross_site_rho2s[train_site_name][test_site_name] = rho2_test

json.dump(cross_site_r2s, open(f"results/run_{n_run}/kan_cross_site_r2_{n_epochs * n_steps}_results.json", "w"))
json.dump(cross_site_rho2s, open(f"results/run_{n_run}/kan_cross_site_rho2_{n_epochs * n_steps}_results.json", "w"))




