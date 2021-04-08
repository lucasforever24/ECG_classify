import os
import numpy as np
import pickle


def create_split(data_dir, datasets=["aortic_sinus"]):
    trainset, valset, testset = [], [], []

    k = 0
    for set in datasets:
        file_lists = os.listdir(os.path.join(data_dir, set))
        ecg_lists = file_lists.copy()

        val_size = len(ecg_lists) // 6
        test_size = len(ecg_lists) // 6
        train_size = len(ecg_lists) - val_size - test_size

        for i in range(0, train_size):
            patient = np.random.choice(ecg_lists)
            ecg_lists.remove(patient)
            trainset.append(str(k) + patient)
        for i in range(0, val_size):
            patient = np.random.choice(ecg_lists)
            ecg_lists.remove(patient)
            valset.append(str(k) + patient)
        for i in range(0, test_size):
            patient = np.random.choice(ecg_lists)
            ecg_lists.remove(patient)
            testset.append(str(k) + patient)

        k = k + 1

    split_dict = dict()
    split_dict['train'] = trainset
    split_dict['val'] = valset
    split_dict['test'] = testset

    with open(os.path.join(data_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)


def create_folds(data_dir, datasets=['aortic_sinus'], folds_num=5):
    folds = []

    for k, set in enumerate(datasets):
        file_lists = os.listdir(os.path.join(data_dir, set))
        copy_lists = file_lists.copy()
        ecg_lists = []
        for item in copy_lists:
            ecg_lists.append(str(k)+item)

        fold_size = len(ecg_lists) // folds_num

        if k == 0:
            for i in range(folds_num - 1):
                subset = []
                for n in range(0, fold_size):
                    patient = np.random.choice(ecg_lists)
                    ecg_lists.remove(patient)
                    subset.append(patient)

                folds.append(subset)

            folds.append(ecg_lists)
        else:
            for i in range(folds_num - 1):
                for n in range(0, fold_size):
                    patient = np.random.choice(ecg_lists)
                    ecg_lists.remove(patient)
                    folds[i].append(patient)

            for item in ecg_lists:
                folds[folds_num-1].append(item)

    splits = []

    for i in range(folds_num):
        split_dcit = dict()
        folds_copy = folds.copy()
        split_dcit['val'] = folds_copy[i]
        folds_copy.remove(folds_copy[i])
        flat_folds = []
        for sublist in folds_copy:
            for item in sublist:
                flat_folds.append(item)
        split_dcit['train'] = flat_folds

        splits.append(split_dcit)

    with open(os.path.join(data_dir, 'folds.pkl'), 'wb') as f:
        pickle.dump(splits, f)


def create_vae_split(data_dir, window_dir):
    trainset, valset, testset = [], [], []

    file_lists = os.listdir(data_dir)
    ecg_lists = file_lists.copy()

    val_size = len(ecg_lists) // 6
    test_size = len(ecg_lists) // 6
    train_size = len(ecg_lists) - val_size - test_size

    for i in range(0, train_size):
        patient = np.random.choice(ecg_lists)
        ecg_lists.remove(patient)
        trainset.append(patient)
    for i in range(0, val_size):
        patient = np.random.choice(ecg_lists)
        ecg_lists.remove(patient)
        valset.append(patient)
    for i in range(0, test_size):
        patient = np.random.choice(ecg_lists)
        ecg_lists.remove(patient)
        testset.append(patient)

    split_dict = dict()
    split_dict['train'] = trainset
    split_dict['val'] = valset
    split_dict['test'] = testset

    with open(os.path.join(window_dir, 'splits.pkl'), 'wb') as f:
        pickle.dump(split_dict, f)



