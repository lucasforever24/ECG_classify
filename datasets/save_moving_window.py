import numpy as np
import os
import glob
import shutil


def save_moving_window(base_dir, datasets, window_size=512, stride=50):
    saving_dir = '../data/moving_windows'
    if not os.path.exists(saving_dir):
        os.mkdir(saving_dir)

    for set in datasets:
        base_set_dir = os.path.join(base_dir, set)
        output_set_dir = os.path.join(saving_dir, set)

        if not os.path.exists(output_set_dir):
            os.mkdir(output_set_dir)

        patients = os.listdir(base_set_dir)
        for ppl in patients:
            patient_dir = os.path.join(output_set_dir, ppl)
            if not os.path.exists(patient_dir):
                os.mkdir(patient_dir)

            ecg_data = []
            leads = os.listdir(os.path.join(base_set_dir, ppl))
            leads.sort()

            for l in leads:
                with open(os.path.join(os.path.join(base_set_dir, ppl), l)) as f:
                    data = f.readlines()
                data = np.asarray(data)
                data = data.astype('float')

                ecg_data.append(data)

            ecg_data = np.asarray(ecg_data)
            # ecg_data = ecg_data/(ecg_data.max() - ecg_data.min())

            save_patient_window(ecg_data, ppl, patient_dir, window_size, stride)


def save_patient_window(data, ppl_name, output_dir, window_size=512, stride=50):
    window_num = data.shape[1] // stride
    for i in range(window_num):
        if (i*stride + window_size) < data.shape[1]:
            segment = data[:, i * stride:(i * stride + window_size)]
            segment = segment / (segment.max() - segment.min())
            saving_path = os.path.join(output_dir, ppl_name + '_' + str(i) + '.npy')
            # np.save(saving_path, segment[0:1])

            print('Saving', ppl_name + '_' + str(i) + '.npy')


if __name__ == "__main__":
    save_moving_window(base_dir='../data/new_data', datasets=['aortic_sinus'])
    base_dir = '../data/moving_windows'
    base_set_dir = os.path.join(base_dir, 'aortic_sinus')
    output_dir = os.path.join(base_dir, 'preprocessed')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    patients = os.listdir(base_set_dir)
    for ppl in patients:
        patient_dir = os.path.join(base_set_dir, ppl)
        segs = os.listdir(os.path.join(base_set_dir, ppl))

        for f in segs:
            shutil.copy(os.path.join(patient_dir, f), os.path.join(output_dir, f))
            print('Copy', f)



