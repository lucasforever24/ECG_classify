from trixi.util import Config
import os


def get_config():
    data_root_dir = os.path.abspath("data")

    c = Config(
        n_epochs=120,
        batch_size=16,
        learning_rate=0.00001,
        beta=1,  # loss normalization parameter

        fold=1,

        print_freq=10,
        do_load_checkpoint=False,
        checkpoint_dir="",

        datasets=["AN", "NV"],
        orig_dir=os.path.join(data_root_dir, "orig_data"),
        new_dir=os.path.join(data_root_dir, 'ecg_data'),
        data_dir=os.path.join(data_root_dir, 'preprocess'),
        result_dir=os.path.abspath("results"),
        split_dir=os.path.join(data_root_dir, 'preprocess')
    )

    print(c)
    return c

