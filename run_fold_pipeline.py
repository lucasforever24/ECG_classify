import os
import datetime
import pickle
import numpy as np

from experiment.fold_experiment import EcgFoldExperiment
from configs.config import get_config
from datasets.preprocess import preprocess_ecg, get_ecg_segments
from datasets.create_splits import create_folds

from utils import AverageMeter, print_metrices_out


if __name__ == "__main__":
    c = get_config()

    if not os.path.exists(c.data_dir):
        os.mkdir(c.data_dir)
        get_ecg_segments(c.new_dir, c.datasets, c.data_dir)
        create_folds(c.data_dir, datasets=c.datasets)

    #  create_folds(c.data_dir, datasets=c.datasets, folds_num=5)

    total_accuracy = AverageMeter()
    total_false_dict = []
    total_predict = []
    total_test = []
    total_prob = []

    now = str(datetime.datetime.now())[:19]
    result_dir = os.path.join(c.result_dir, 'ecg_fold_' + now)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print('Creating result dir:', result_dir)
    c.result_dir = result_dir

    # load the splits file for only once to avoid conflicts with other experiment
    with open(os.path.join(c.split_dir, "folds.pkl"), 'rb') as f:
        splits = pickle.load(f)

    for k in range(5):
        c.fold = k
        experiment = EcgFoldExperiment(c, splits)

        for i in range(c.n_epochs):
            experiment.train(i)
            experiment.val(i)

        accuracy, false_dict, metrics = experiment.test()

        total_accuracy.update(accuracy.avg, accuracy.count)
        total_false_dict.append(false_dict)
        total_predict.append(metrics['predict'])
        total_test.append(metrics['test'])
        total_prob.append(metrics['prob'])

    total_predict = np.concatenate(total_predict)
    total_prob = np.concatenate(total_prob)
    total_test = np.concatenate(total_test)

    print_metrices_out(total_predict, total_test, total_prob)
    print("Total accuracy:", total_accuracy.avg)
    print(total_false_dict)


