import os
import matplotlib.pyplot as plt
import numpy as np


def save_ecg(ecg, patient_dir):
    figure = plt.figure(figsize=(12, 8))
    axes1 = figure.add_subplot(4, 3, 1)
    axes2 = figure.add_subplot(4, 3, 2)
    axes3 = figure.add_subplot(4, 3, 3)
    axes4 = figure.add_subplot(4, 3, 4)
    axes5 = figure.add_subplot(4, 3, 5)
    axes6 = figure.add_subplot(4, 3, 6)
    axes7 = figure.add_subplot(4, 3, 7)
    axes8 = figure.add_subplot(4, 3, 8)
    axes9 = figure.add_subplot(4, 3, 9)
    axes10 = figure.add_subplot(4, 3, 10)
    axes11 = figure.add_subplot(4, 3, 11)
    axes12 = figure.add_subplot(4, 3, 12)

    axes1.plot(ecg[0])
    axes2.plot(ecg[1])
    axes3.plot(ecg[2])
    axes4.plot(ecg[3])
    axes5.plot(ecg[4])
    axes6.plot(ecg[5])
    axes7.plot(ecg[6])
    axes8.plot(ecg[7])
    axes9.plot(ecg[8])
    axes10.plot(ecg[9])
    axes11.plot(ecg[10])
    axes12.plot(ecg[11])

    plt.savefig(os.path.join(patient_dir, "12leads.png"))


if __name__ == "__main__":
    base_dir = '/Users/lucasforever24/PycharmProjects/ECG_classify/data/orig_data/normal2'
    output_dir = '/Users/lucasforever24/PycharmProjects/ECG_classify/data/patients_fig'
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    patients = os.listdir(base_dir)
    for ppl in patients:
        patient_dir = os.path.join(base_dir, ppl)

        leads = os.listdir(patient_dir)

        data = []

        if len(leads) != 0:
            for k in range(len(leads)):
                with open(os.path.join(patient_dir, leads[k])) as f:
                    lead_data = f.readlines()

                lead_data = np.asarray(lead_data)
                lead_data = lead_data.astype('float')

                data.append(lead_data)

            data = np.asarray(data)

            save_path = os.path.join(output_dir, ppl)
            if not os.path.exists(save_path):
                os.mkdir(save_path)

            save_ecg(data, save_path)
            plt.close()

            print(save_path)


