import os
import datetime
import shutil
import pickle
import numpy as np

from experiment.fold_seg_experiment import EcgFoldSegExperiment
from configs.config_unet import get_config
from utils import save_dict_to_txt
from datasets.preprocess import preprocess_ecg, get_ecg_segments
from datasets.create_splits import create_folds

from utils import AverageMeter, print_metrices_out, print_metrices_out_multiclass


def train_and_val(config):

    # create_folds(config.data_dir, datasets=config.datasets, folds_num=5)

    total_accuracy = AverageMeter()
    total_predict = []
    total_test = []
    total_prob = []

    # build result directory
    now = str(datetime.datetime.now())[:19]
    result_dir = os.path.join(config.result_dir, 'ecg_seg_fold_' + now)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print('Creating result dir:', result_dir)
    config.result_dir = result_dir

    with open(os.path.join(config.split_dir, "folds.pkl"), 'rb') as f:
        splits = pickle.load(f)

    total_false_dict = dict()
    total_score_dict = dict()

    for k in range(5):
        c.fold = k
        experiment = EcgFoldSegExperiment(config, splits)

        for i in range(config.n_epochs):
            experiment.train(i)
            experiment.val(i)
            if experiment.loss_check():
                continue

        accuracy, false_dict, score_dict, metrics = experiment.test()

        total_accuracy.update(accuracy.avg, accuracy.count)
        total_false_dict.update(false_dict)
        total_score_dict.update(score_dict)
        total_predict.append(metrics['predict'])
        total_test.append(metrics['test'])
        total_prob.append(metrics['prob'])

    total_predict = np.concatenate(total_predict)
    total_prob = np.concatenate(total_prob)
    total_test = np.concatenate(total_test)

    save_dict_to_txt(total_false_dict, os.path.join(result_dir, "false_dict.txt"))
    save_dict_to_txt(total_score_dict, os.path.join(result_dir, "score_dict.txt"))

    print_metrices_out_multiclass(total_predict, total_test, total_prob)
    print("Total accuracy:", total_accuracy.avg)
    print(total_false_dict)


def inference(config):
    # build result directory
    now = str(datetime.datetime.now())[:19]
    result_dir = os.path.join(config.result_dir, 'seg_inference_' + now)
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
        print('Creating result dir:', result_dir)
    config.result_dir = result_dir

    config.fold = 0
    config.do_load_checkpoint = True
    config.checkpoint_dir = 'results/ecg_seg_fold_2020-10-02 00:41:57/2020-10-02 00:42:00'

    # load the splits file for only once to avoid conflicts with other experiment
    with open(os.path.join(config.split_dir, "folds.pkl"), 'rb') as f:
        splits = pickle.load(f)
    experiment = EcgFoldSegExperiment(config, splits)

    experiment.inference()


if __name__ == "__main__":
    c = get_config()
    c.model = 'u'

    # inference(c)
    train_and_val(c)





