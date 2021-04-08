from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from torchvision import transforms


class EcgDataset(Dataset):
    def __init__(self, data_dir, keys, datasets):
        self.data_dir = data_dir
        self.keys = keys
        self.datasets = datasets
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.files = []
        self.labels = []
        self.fname = []
        for patient in keys:
            label = int(patient[0])
            location_dir = os.path.join(data_dir, datasets[label])
            patient_dir = os.path.join(location_dir, patient[1:])
            for f in os.listdir(patient_dir):
                self.files.append(os.path.join(patient_dir, f))
                self.labels.append(label)
                self.fname.append(f)

    def __getitem__(self, index):
        file_name = self.fname[index]
        file_path = self.files[index]
        ecg_data = np.load(file_path)
        ecg_data = ecg_data
        ecg_data = ecg_data.astype("float")
        label = self.labels[index]

        data = {'ecg': ecg_data, 'label': label, 'fname': file_name}

        return data

    def __len__(self):
        return len(self.files)


class VaeDataset(Dataset):
    def __init__(self, data_dir, keys):
        self.data_dir = data_dir
        self.keys = keys
        self.transform = transforms.Compose([transforms.ToTensor()])

        self.files = []
        self.labels = []
        for f in keys:
            self.files.append(os.path.join(data_dir, f))

    def __getitem__(self, index):
        file_path = self.files[index]
        ecg_data = np.load(file_path)

        data = {'ecg': ecg_data}

        return data

    def __len__(self):
        return len(self.files)


