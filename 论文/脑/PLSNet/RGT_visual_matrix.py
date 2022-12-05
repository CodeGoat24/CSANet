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
matrix_pearson = []
matrix_attention = []
scores = []
scores_all = []

def test_score(dataloader, model):
    result = []
    sc = []



    for data_in, pearson, label, pseudo in dataloader:
        label = label.long()
        data_in, pearson, label, pseudo = data_in.to(
            device), pearson.to(device), label.to(device), pseudo.to(device)
        [output, score, cor_matrix], matrix, _ = model(data_in, pearson, pseudo)

        result += F.softmax(output, dim=1)[:, 1].tolist()

        global matrix_pearson
        matrix_pearson += data_in.detach().cpu().numpy().tolist()
        global matrix_attention
        matrix_attention += matrix.detach().cpu().numpy().tolist()

        cor = cor_matrix[:, :, 0, :].to(device)
        # 计算score
        score = cor[:, 0, :]
        for i in range(3):
            score += cor[:, i + 1, :]

        score = score[:, 1:]
        sc += score.detach().cpu().numpy().tolist()
        global scores_all
        scores_all += score.detach().cpu().numpy().tolist()

    result = np.array(result)
    result[result > 0.5] = 1
    result[result <= 0.5] = 0
    sc = np.array(sc)
    global scores
    scores += sc[result == 1].tolist()



with open('setting/abide_fbnetgen.yaml') as f:
    config = yaml.load(f, Loader=yaml.Loader)
    (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries_size = init_dataloader(config['data'])
    # print(config['data']['batch_size'])
    model = GraphTransformer(config['model'], node_size,
                 node_feature_size, timeseries_size).to(device)
    # model.load_state_dict(torch.load('/home/star/CodeGoat24/FBNETGEN/result/GraphTransformer/AAL_70.936%/model_70.93596055354978%.pt'))
    model.load_state_dict(torch.load('/home/star/CodeGoat24/FBNETGEN/result/GraphTransformer/ABIDE_aal_gcn4/ 73.399%_09-11-03-44-22/model_73.3990148910748%.pt'))

    torch.cuda.set_device(0)

    model.eval()
    test_score(dataloader=train_dataloader, model=model)
    test_score(dataloader=val_dataloader, model=model)
    test_score(dataloader=test_dataloader, model=model)

    csv_reader = csv.reader(open('aal_labels2.csv', encoding='utf-8'))
    label = [row[1] for row in csv_reader]
    # 画Pearson邻接矩阵
    matrix_pearson = np.array(matrix_pearson)

    connectivity = ConnectivityMeasure(kind='correlation')
    connectivity_matrices = connectivity.fit_transform(matrix_pearson.swapaxes(1, 2))
    mean_connectivity_matrices = connectivity_matrices.mean(axis=0)

    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=1.2, vmin=-0.05, colorbar=True, reorder=False, title="")
    plotting.show()


    # 画attention邻接矩阵
    matrix_attention = np.array(matrix_attention)


    mean_connectivity_matrices = matrix_attention.mean(axis=0)
    plotting.plot_matrix(mean_connectivity_matrices, figure=(10, 8), labels=label[2:], vmax=0.8758, vmin=0.862,
                         colorbar=True,
                         reorder=False, title="")

    plotting.show()