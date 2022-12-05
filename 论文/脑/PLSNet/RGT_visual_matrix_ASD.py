import csv

import numpy
from nilearn.connectome import ConnectivityMeasure
from nilearn import plotting
import yaml
import torch
import os


from model.GraphTransformer import GraphTransformer

from dataloader import init_dataloader
from util import Logger, accuracy, TotalMeter
import numpy as np

import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
# device = torch.device("cpu")

count = {}
matrix_con = []
matrix_asd = []
scores = []
scores_all = []

def test_score(dataloader, model):
    result = []
    matrix_all= []



    for data_in, pearson, label, pseudo in dataloader:
        label = label.long()
        data_in, pearson, label, pseudo = data_in.to(
            device), pearson.to(device), label.to(device), pseudo.to(device)
        [output, score, cor_matrix], matrix, _ = model(data_in, pearson, pseudo)

        result += F.softmax(output, dim=1)[:, 1].tolist()
        matrix_all += matrix.detach().cpu().numpy().tolist()


    result = np.array(result)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    matrix_all = np.array(matrix_all)
    global matrix_asd
    matrix_asd += matrix_all[result == 1].tolist()
    global matrix_con
    matrix_con += matrix_all[result == 0].tolist()



with open('setting/abide_fbnetgen.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    # print(config['data']['batch_size'])
    model = GraphTransformer(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    # model.load_state_dict(torch.load('/home/star/CodeGoat24/FBNETGEN/result/GraphTransformer/AAL_70.936%/model_70.93596055354978%.pt'))
    model.load_state_dict(torch.load('/home/star/CodeGoat24/FBNETGEN/result/GraphTransformer/ABIDE_aal_gcn4/ 73.399%_09-11-03-44-22/model_73.3990148910748%.pt'))


    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)

    csv_reader = csv.reader(open('aal_labels2.csv', encoding='utf-8'))
    label = [row[1] for row in csv_reader]
    # 画asd邻接矩阵
    matrix_asd = np.array(matrix_asd)

    mean_connectivity_matrices = matrix_asd.mean(axis=0)
    np.save('./asd.npy', mean_connectivity_matrices)

    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8643, vmin=0.848, colorbar=True, reorder=False, title="")
    plotting.show()


    # 画con邻接矩阵
    matrix_con = np.array(matrix_con)

    mean_connectivity_matrices = matrix_con.mean(axis=0)
    np.save('./health.npy', mean_connectivity_matrices)
    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8965, vmin=0.873,
                         colorbar=True,
                         reorder=False, title="")
    plotting.show()