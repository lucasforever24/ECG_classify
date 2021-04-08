import scipy.io
import wfdb
import os
import numpy as np
import pickle


def segment_record(record, r_peaks, label, fix_length=2000):
    num_peaks = len(r_peaks)
    start_peak = 1
    end_peak = 1
    segments = []
    label_list = []
    total_label_point = []
    while start_peak < num_peaks:
        label_point = []
        segment_length = 0
        s_point = np.ceil(r_peaks[start_peak] / 3 + 2 * r_peaks[start_peak - 1] / 3)
        segment_label = []

        segment_label.append(label[start_peak])
        if label[start_peak] != 'N':
            label_point.append(r_peaks[start_peak])

        while segment_length < 2000:
            end_peak += 1
            if label[end_peak] not in segment_label:
                segment_label.append(label[end_peak])

            if label[start_peak] != 'N':
                label_point.append(r_peaks[start_peak])

            if end_peak == num_peaks - 1:
                break
            e_point = np.ceil(r_peaks[end_peak + 1] / 3 + 2 * r_peaks[end_peak] / 3)
            segment_length = e_point - s_point

        if end_peak == num_peaks - 1:
            break

        e_point = np.ceil(r_peaks[end_peak] / 3 + 2 * r_peaks[end_peak - 1] / 3)
        s_point = s_point.astype('int')
        e_point = e_point.astype('int')
        if (e_point - s_point) < 2000:
            segment = record[:, s_point:(e_point + 1)]
            segment = padding_zeros(segment, fix_length)
            segments.append(segment)
            segment_label.sort()
            segment_label = "".join(segment_label)
            label_list.append(segment_label)

        start_peak = end_peak
        total_label_point.append(label_point)

    return segments, label_list, total_label_point


def padding_zeros(segment, target_length):
    new_segment  =  np.zeros((segment.shape[0], target_length))
    before = (target_length - segment.shape[1]) // 2
    after = target_length - segment.shape[1] - before
    for k in range(segment.shape[0]):
        new_segment[k] = np.pad(segment[0], (before, after), 'constant', constant_values=(0, 0))

    assert new_segment.shape[1] == target_length

    return new_segment


def get_number_list(base_dir):
    file_list = [f.split('.')[0] for f in os.listdir(base_dir) if f[0].isdigit()]
    number_list = []
    for file in file_list:
        if file not in number_list:
            number_list.append(file)

    number_list.sort()
    return number_list


def save_segment(base_dir, data_dir, label_dir, considered_name=[]):
    # read wfdb form file in base_dir and save them in data_dir
    number_list = get_number_list(base_dir)
    # ecg_dir = os.path.join(data_dir, 'data')
    # label_dir = os.path.join(data_dir, 'label')
    for name in considered_name:
        os.makedirs(os.path.join(data_dir, name), exist_ok=True)

    for f in number_list:
        record = wfdb.rdsamp(os.path.join(base_dir, f))[0]  # the original shape is (650000, 2)
        record = np.transpose(record, axes=[1, 0])
        annotation = wfdb.rdann(os.path.join(base_dir, f), 'atr')

        r_peaks = annotation.sample
        label = annotation.symbol

        new_peaks, new_label = remove_non_beat_annotation(r_peaks, label)
        segments, label_list, total_label_point = segment_record(record, new_peaks, new_label)

        j = 0
        for i, segment in enumerate(segments):
            segment_label = label_list[i]
            if segment_label in considered_name:
                output_folder = os.path.join(data_dir, segment_label)
                label_folder = os.path.join(label_dir, segment_label)
                # np.save(os.path.join(output_folder, f + '_' + str(i) + '.npy'), segment)
                np.save(os.path.join(label_folder, f + '_' + str(i) + "_label" + '.npy'), total_label_point[i])
                j += 1

        print('Finished saving %d segments of %s' % (j, f))



def calculate_segment_classes(base_dir):
    number_list = get_number_list(base_dir)
    label_dict = dict()

    for f in number_list:
        print('Processing', f)
        record = wfdb.rdsamp(os.path.join(base_dir, f))[0]  # the original shape is (650000, 2)
        record = np.transpose(record, axes=[1, 0])
        annotation = wfdb.rdann(os.path.join(base_dir, f), 'atr')

        r_peaks = annotation.sample
        label = annotation.symbol

        new_peaks, new_label = remove_non_beat_annotation(r_peaks, label)

        segments, label_list = segment_record(record, new_peaks, new_label)
        # print(label_list)

        for seg_label in label_list:
            seg_label = "".join(seg_label)
            if seg_label not in label_dict.keys():
                label_dict[seg_label] = 1
            else:
                label_dict[seg_label] += 1

        print(label_dict)

    with open('label_dict.pickle', 'wb') as f:
        pickle.dump(label_dict, f)



def remove_non_beat_annotation(r_peaks, label):
    beat_ann_list = ['/', 'A', 'E', 'F', 'J', 'L', 'N', 'Q', 'R', 'S', 'V', 'a', 'e',
                     'f', 'j']
    new_peaks = []
    new_label = []
    for i, ann in enumerate(label):
        if ann in beat_ann_list:
            new_peaks.append(r_peaks[i])
            new_label.append(label[i])

    return new_peaks, new_label


if __name__ == "__main__":
    base_dir = "../../data/mitdb"
    data_dir = "../../data/preprocess"
    label_dir = "../../data/label"

    save_segment(base_dir, data_dir, label_dir, considered_name=['AN', 'NV'])
    # calculate_segment_classes(base_dir)
