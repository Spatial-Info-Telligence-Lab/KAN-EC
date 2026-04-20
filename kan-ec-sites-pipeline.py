import torch.nn.functional as F
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt

import pandas as pd

site_name = "xHA_clean"

feature_dict = {
    'temperature': 'TA_F',
    'VPD': 'VPD_F',
    'soil moisture': 'SWC_F_MDS_1',
    'precipitation': 'P_F',
    'soil temperature': 'TS_F_MDS_1',
    'shortwave radiation': 'SW_IN_F',
}

n_epochs = 10
n_steps = 10

import os

directory_path = './data/site_data' # Use '.' for the current directory or specify your path

all_entries = os.listdir(directory_path)

site_files = []
for entry in all_entries:
    root, extension = os.path.splitext(entry)
    full_path = os.path.join(directory_path, entry)
    if extension == ".csv" and os.path.join(directory_path, entry):
        site_files.append(full_path)

mlp_results = pd.read_csv("data/site_data/mlp_results.csv")

from kan import *
torch.set_default_dtype(torch.float64)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).
input_dim = len(feature_dict)

df = pd.read_csv(f"./data/site_data/{site_name}.csv")

features = list(feature_dict.values())
target_col = 'RECO_NT_VUT_REF'

# Drop rows with NaNs in features or target
df_model = df.dropna(subset=features + [target_col, 'year']).copy()

years = sorted(df_model['year'].unique())
print("Years in dataset:", years)

yearly_results = {}

for test_year in years:
    print(f"\n=== Fold: Test year = {test_year} ===")

    train_df = df_model[df_model['year'] != test_year]
    test_df = df_model[df_model['year'] == test_year]

    train_input = train_df[features].values
    train_label = np.log(train_df[target_col].values.reshape(-1, 1))

    test_input = test_df[features].values
    test_label = np.log(test_df[target_col].values.reshape(-1, 1))

    dataset = {
        "train_input": torch.DoubleTensor(train_input).to(device),
        "train_label": torch.DoubleTensor(train_label).to(device),
        "test_input": torch.DoubleTensor(test_input).to(device),
        "test_label": torch.DoubleTensor(test_label).to(device)
    }

    print(dataset['train_input'].shape, dataset['train_label'].shape)
    print(dataset['test_input'].shape, dataset['test_label'].shape)

    # KAN initialization
    # model = KAN(width=[input_dim, 2 * input_dim + 1, 1], grid=3, k=3, noise_scale=0.1, seed=1, device=device)
    model = KAN(width=[input_dim, 2 * input_dim + 1, 1], grid=3, k=3, noise_scale=0.3, seed=1, device=device)

    # train the model
    model.train()
    for e in range(n_epochs):
        print(f"Training Epoch {e+1} Starts")
        model.fit(dataset, opt="LBFGS", steps=n_steps, lamb=0.00, lamb_entropy=0.0, lr=1)
        # print(f"KAN visualization for test year {test_year}")

        model.eval()
        with torch.no_grad():
            train_pred = model(dataset['train_input']).cpu().numpy().ravel()
            test_pred = model(dataset['test_input']).cpu().numpy().ravel()

            r2_train = r2_score(np.exp(train_label), np.exp(train_pred))
            r2_test = r2_score(np.exp(test_label), np.exp(test_pred))

        print(f"Evaluation for Epoch {e+1}. Test year {test_year}: Train R² {r2_train:.3f}, Test R² {r2_test:.3f}")
        print(f"Training Epoch {e + 1} Ends")

    # model.prune()
    #
    # model.fit(dataset, opt="LBFGS", steps=n_steps, lamb=0.0, lamb_entropy=0.0, lr=1.0, update_grid=False)
    #
    # model.eval()
    # with torch.no_grad():
    #     train_pred = model(dataset['train_input']).cpu().numpy().ravel()
    #     test_pred = model(dataset['test_input']).cpu().numpy().ravel()
    #
    #     r2_train = r2_score(np.exp(train_label), np.exp(train_pred))
    #     r2_test = r2_score(np.exp(test_label), np.exp(test_pred))
    #
    # print(f"Evaluation. Test year {test_year}: Train R² {r2_train:.3f}, Test R² {r2_test:.3f}")

    yearly_results[int(test_year)] = {"train": float(r2_train), "test": float(r2_test)}


r2_train_list = []
r2_test_list = []
for r in yearly_results.values():
    r2_train_list.append(r["train"])
    r2_test_list.append(r["test"])

# print(f"MLP Average: Train R² {mlp_results[mlp_results['site_ID']=='US-xST'].mean_r2_train:.3f}, Test R² {mlp_results[mlp_results['site_ID']=='US-xST'].mean_r2_test:.3f}")
print(f"KAN Average: Train R² {np.mean(r2_train_list):.3f}, Test R² {np.mean(r2_test_list):.3f}")

import json

json.dump(yearly_results, open(f"results/kan_{site_name}_{n_steps}_results.json", "w"))

