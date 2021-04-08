import os
import numpy as np
import torch
import torch.nn.functional as F

from datasets.ecg_utils import lp_filter, baseline_fix, get_abnormal_r_peaks
from networks.AnomalyClassifier import AnomalyClassifier
from networks.segment.ECGUNet import ECGUNet

import shutil


def inference(data_dir, checkpoint_dir):
    # preprocess ecg
    patients = [f for f in os.listdir(data_dir) if not f.startswith(".")]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load checkpoints
    # model = AnomalyClassifier(in_channels=12, num_classes=2)
    model = ECGUNet(in_channels=12, num_classes=2, unet_classes=1)
    model.load_state_dict(torch.load(checkpoint_dir, map_location=device))
    model.to(device)

    batch_data = []
    for ppl in patients:
        print("========================")
        print('Processing ecg of', ppl)

        patient_dir = os.path.join(data_dir, ppl)
        leads = [f for f in os.listdir(patient_dir) if not f.startswith(".")]
        ecg_data = []
        leads.sort()
        for s in leads:
            data_path = os.path.join(patient_dir, s)
            with open(data_path) as f:
                lines = f.readlines()

            data = np.asarray(lines).astype('float')

            data1 = lp_filter(1000, 20, data)
            data2 = baseline_fix(1000, data1)

            ecg_data.append(data2)

        ecg_data = np.asarray(ecg_data)
        print(ecg_data.shape)
        batch_data.append(ecg_data)

        """
        r_peaks = get_abnormal_r_peaks(ecg_data)

        # get beats segmentation
        for i, point in enumerate(r_peaks):
            segment = segment_ecg(ecg_data, point)
            if segment is not None:
                assert segment.shape[0] == 12, ppl
                assert segment.shape[1] == 500, ppl
                segment = segment / (segment.max() - segment.min())
                batch_data.append(segment)

        print('Number of segment this patient:', len(batch_data))
        """

    batch_data = np.asarray(batch_data)
    batch_data = torch.Tensor(batch_data)
    if len(batch_data.shape) != 3:
        batch_data = batch_data.unsqueeze(0)
    print("Preprocess finished")
    print(batch_data.shape)

    data = batch_data.float().to(device)
    output = model(data)
    output_softmax = F.softmax(output[1], dim=1)
    prediction = torch.argmax(output_softmax, dim=1)

    print_prediction(prediction, patients)


def print_prediction(prediction, patients):
    assert prediction.shape[0] == len(patients), 'Something wrong with the batch number'
    for k in range(len(patients)):
        if prediction[k] == 0:
            print('The PVC origin of patient %s is from left' % patients[k])
        else:
            print('The PVC origin of patient %s is from right' % patients[k])



def segment_ecg(data, r_peak, left_margin=200, right_margin=300):
    start_point = r_peak - left_margin
    end_point = r_peak + right_margin
    if start_point > 200 and end_point < len(data[0]):
        return data[:, start_point:end_point]
    else:
        return None

def rearrange_lead(base_dir):
    # make sure for each patient dir, there are only 12 leads;
    patients = [f for f in os.listdir(base_dir) if not f.startswith(".")]
    for ppl in patients:
        patient_dir = os.path.join(base_dir, ppl)
        leads = [f for f in os.listdir(patient_dir) if not f.startswith(".")]
        leads.sort()
        assert (len(leads) % 12) == 0, ppl
        groups = len(leads) // 12
        for k in range(groups):
            if k > 0:
                dst_dir = os.path.join(base_dir, ppl + str(k))
                if not os.path.exists(dst_dir):
                    os.mkdir(dst_dir)
                print("Making new directory:", ppl + str(k))
                for m in range(12):
                    index = k * 12 + m
                    shutil.move(os.path.join(patient_dir, leads[index]), os.path.join(dst_dir, leads[index]))


if __name__ == "__main__":

    data_dir = "/Users/lucasforever24/PycharmProjects/ECG_classify/data/inference/"
    checkpoint_dir = "/Users/lucasforever24/PycharmProjects/ECG_classify/" \
                     "results/2020-10-02 01:47:40/model.pth"

    rearrange_lead(data_dir)
    inference(data_dir, checkpoint_dir)

