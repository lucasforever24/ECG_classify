import os

from experiment.experiment import EcgExperiment
from configs.config import get_config
from datasets.preprocess import preprocess_ecg, get_ecg_segments
from datasets.create_splits import create_split


if __name__ == "__main__":
    c = get_config()
    if not os.path.exists(c.data_dir):
        os.mkdir(c.data_dir)
    else:
        print("Data have already been preprocessed")

    create_split(c.data_dir, c.datasets)

    experiment = EcgExperiment(c)

    for k in range(c.n_epochs):
        experiment.train(k)
        experiment.val(k)

    experiment.test()

