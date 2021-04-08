import os

from experiment.vae.vae_experiment import VaeExperiment
from configs.config_vae import get_config
from datasets.preprocess import preprocess_ecg
from datasets.create_splits import create_vae_split

if __name__ == "__main__":
    c = get_config()

    if not os.path.exists(c.data_dir):
        preprocess_ecg(c.orig_dir, c.new_dir)
    else:
        print("Data have already been preprocessed")

    create_vae_split(c.data_dir, c.window_dir)

    if not os.path.exists(c.result_dir):
        os.mkdir(c.result_dir)
        print('Creating result dir:', c.result_dir)

    experiment = VaeExperiment(c)

    for i in range(100):
        experiment.train(i)
        experiment.val(i)

    experiment.test()



