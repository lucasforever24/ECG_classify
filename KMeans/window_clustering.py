import numpy as np
import os
import pickle
import torch
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans

import sys
sys.path.append('../')

from networks.vae.VAE_1d import VAE


if __name__  == "__main__":
    ####### prepare data, calculate latent embeddings for every window #######
    state_dict_path = '../results/' + '2020-08-25 22:19:10/' + 'model.pth'
    data_dir = '../data/moving_windows/preprocessed'
    test_dir = '../data/moving_windows/aortic_sinus/P900243李燕葵'

    model = VAE(in_channels=1, latent_dim=32, batch_size=1)
    model.load_state_dict(torch.load(state_dict_path, map_location=torch.device('cpu')))
    model.eval()

    data = dict()
    data['fname'] = []
    data['x'] = []

    file_list = os.listdir(data_dir)
    file_list.sort()
    for f in file_list:
        data['fname'].append(f)
        ecg_signal = np.load(os.path.join(data_dir, f))
        ecg_signal = torch.Tensor(ecg_signal).view(1, 1, -1)
        with torch.no_grad():
            mu, log_var = model.encode(ecg_signal)
            embedding = torch.exp(0.5*log_var) + mu
        data['x'].append(embedding.squeeze().numpy())

    kf = KFold(n_splits=9, shuffle=False)

    X = np.asarray(data['x'])
    fname = np.asarray(data['fname'])

    for train_index, test_index in kf.split(data['fname']):
        x_train, x_test = X[train_index], X[test_index]
        fname_train, fname_test = fname[train_index], fname[test_index]
        continue

    print('Train data size:', x_train.shape, 'Test data size:', x_test.shape)

    ######### perform k clustering ##########
    kmeans = KMeans(n_clusters=2, random_state=0).fit(x_train[0:90])
    predict_label = kmeans.predict(x_train[0:90])
    print(predict_label)
    print(fname_train[0:90])
    print(kmeans.cluster_centers_)

    predict_result = {'label': predict_label, 'fname': fname_teste}

    with open('predict_result.p', 'wb') as fp:
        pickle.dump(predict_result, fp)

