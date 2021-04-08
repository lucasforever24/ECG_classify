import os
import numpy as np
import shutil

import sys
sys.path.append('../')

from datasets.ecg_utils import lp_filter, baseline_fix, get_abnormal_r_peaks


def preprocess_ecg(orig_dir, output_dir):
    # preprocess ecg data in the orig_dir
    patients = os.listdir(orig_dir)

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    total = 0
    for ppl in patients:
        print('Processing ecg of', ppl)

        new_patient = os.path.join(output_dir, ppl)
        if not os.path.exists(new_patient):
            os.mkdir(new_patient)

        patient_dir = os.path.join(orig_dir, ppl)
        ecg_data = []
        leads = os.listdir(patient_dir)
        for s in leads:
            data_path = os.path.join(patient_dir, s)
            with open(data_path) as f:
                lines = f.readlines()

            data = np.asarray(lines).astype('float')

            data1 = lp_filter(1000, 20, data)
            data2 = baseline_fix(1000, data1)

            np.savetxt(os.path.join(new_patient, s), data2)

            total += 1

        print('Saving new', ppl)

    print("Preprocess finished!", total)


def get_ecg_segments(data_dir, datasets, dst_dir, delete_old_data=True):
    # store selected segments in a lead in npy file, eg. data/preprocessed/aortic_sinus/Pxxxxxx/xxx.npy
    # notice that a patient may have several directories because he/she has more than 12 leads records
    # first load 12 leads and then find the segments you need

    for set in datasets:
        output_dir = os.path.join(dst_dir, set)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif delete_old_data:
            shutil.rmtree(output_dir)
            print('Removing previous data...')
            os.mkdir(output_dir)

        set_dir = os.path.join(data_dir, set)
        patients = os.listdir(set_dir)
        for ppl in patients:
            ecg_data = []
            leads = os.listdir(os.path.join(set_dir, ppl))
            leads.sort()

            for l in leads:
                with open(os.path.join(os.path.join(set_dir, ppl), l)) as f:
                    data = f.readlines()
                data = np.asarray(data)
                data = data.astype('float')

                ecg_data.append(data)

            ecg_data = np.asarray(ecg_data)
            r_peaks = get_abnormal_r_peaks(ecg_data)

            patient_dst_dir = os.path.join(output_dir, ppl)
            if not os.path.exists(patient_dst_dir):
                os.mkdir(patient_dst_dir)

            for i, point in enumerate(r_peaks):
                segment = segment_ecg(ecg_data, point)
                if segment is not None:
                    segment = segment / (segment.max() - segment.min())
                    store_path = os.path.join(patient_dst_dir, ppl + "_aug" + str(i) + '.npy')
                    np.save(store_path, segment)
                    print("Saving ", ppl + "_" + str(i))

        print("%s saving finished" %(set))


def segment_ecg(data, r_peak, left_margin=200, right_margin=300):
    start_point = r_peak - left_margin
    end_point = r_peak + right_margin
    if start_point > 200 and end_point < len(data[0]):
        return data[:, start_point:end_point]
    else:
        return None


def rearrange_lead(base_dir):
    # make sure for each patient dir, there are only 12 leads;
    patients = os.listdir(base_dir)
    for ppl in patients:
        patient_dir = os.path.join(base_dir, ppl)
        leads = os.listdir(patient_dir)
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


def save_ecg_as_array(data_dir, datasets, dst_dir, delete_old_data=True):
    for set in datasets:
        output_dir = os.path.join(dst_dir, set)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        elif delete_old_data:
            shutil.rmtree(output_dir)
            print('Removing previous data...')
            os.mkdir(output_dir)

        set_dir = os.path.join(data_dir, set)
        patients = os.listdir(set_dir)
        for ppl in patients:
            ecg_data = []
            leads = os.listdir(os.path.join(set_dir, ppl))
            leads.sort()

            for l in leads:
                with open(os.path.join(os.path.join(set_dir, ppl), l)) as f:
                    data = f.readlines()
                data = np.asarray(data)
                data = data.astype('float')

                ecg_data.append(data)

            ecg_data = np.asarray(ecg_data)
            ecg_data = ecg_data / (ecg_data.max() - ecg_data.min())

            patient_dst_dir = os.path.join(output_dir, ppl)
            if not os.path.exists(patient_dst_dir):
                os.mkdir(patient_dst_dir)

            store_path = os.path.join(patient_dst_dir, ppl + '.npy')
            np.save(store_path, ecg_data)

            print("Saving ", ppl)

        print("%s saving finished" % (set))


def normalize_ecg(data_dir, datasets):
    for set in datasets:
        set_dir = os.path.join(data_dir, set)
        patients = os.listdir(set_dir)
        for ppl in patients:
            data_path = os.listdir(os.path.join(set_dir, ppl))[0]
            ecg_data = np.load(os.path.join(set_dir, ppl, data_path))

            for k in range(ecg_data.shape[0]):
                if ecg_data[k].max() != ecg_data[k].min():
                    ecg_data[k] = ecg_data[k] / (ecg_data[k].max() - ecg_data[k].min())

            data1 = lp_filter(360, 20, ecg_data)
            data2 = baseline_fix(360, data1)

            np.save(os.path.join(set_dir, ppl, data_path), data2)

            print("Normalizing ", ppl)

        print("%s normalization finished" %(set))



if __name__ == "__main__":

    base_dir = '../data/ecg_data'
    data_dir = '../data/orig_data/lv_papillary'
    ecg_dir = '../data/ecg_data/lv_papillary'
    target_dir = '../data/preprocess'
    # reference_path = '/Users/lucasforever24/PycharmProjects/ECG_classify/data/orig_data/normal_filter/纳入'

    # rearrange_lead(data_dir)
    # preprocess_ecg(data_dir, ecg_dir)
    # save_ecg_as_array(base_dir, ['left_outflow', 'right_outflow', 'TA', 'lv_papillary'], target_dir)
    normalize_ecg(target_dir,  ['left_outflow', 'right_outflow', 'TA', 'lv_papillary'])



    """
    for f in os.listdir(base_dir):
        new_folder = os.path.join(base_dir, f.split('.')[0])
        os.mkdir(new_folder)
        shutil.move(os.path.join(base_dir, f),  os.path.join(new_folder, f))
    # rearrange the directory so that each diretory has 12 leads
    base_dir = '/Users/lucasforever24/PycharmProjects/ECG_classify/data/orig_data/normal2'

    # preprocess_ecg(data_path, output_dir)

    get_ecg_segments(data_path, datasets=['aortic_sinus', 'pulmonary_sinus'], dst_dir=output_dir, delete_old_data=False)
    """
                    










