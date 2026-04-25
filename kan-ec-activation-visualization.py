import torch.nn.functional as F
from sklearn.metrics import r2_score
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

node_th, edge_th = 5e-2, 5e-2

import os

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

for site_file in site_files:
    print(site_file)
    site_name = site_file.split('\\')[-1].split('_')[0]

    df = pd.read_csv(site_file)

    # Drop rows with NaNs in features or target
    df_model = df.dropna(subset=features + [target_col, 'year']).copy()

    years = sorted(df_model['year'].unique())
    print(f"Years in dataset {site_name}: {years}")

    for test_year in years:
        # Load KAN checkpoint
        model = KAN.loadckpt(f"./model/run_{n_run}/kan_{site_name}_{test_year}_{n_epochs * n_steps}_model")
        model = model.prune(node_th=node_th, edge_th=edge_th)
        model.plot(beta=5, scale=2.0, in_vars=list(feature_dict.values()), out_vars=['RECO'], varscale=0.2)

        plt.savefig(f"./plots/run_{n_run}/kan_{site_name}_{test_year}_{n_epochs * n_steps}_activation.png")