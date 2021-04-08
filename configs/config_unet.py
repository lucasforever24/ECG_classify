from trixi.util import Config
import os


def get_config():
    data_root_dir = os.path.abspath("data")

    c = Config(
        n_epochs=150,
        batch_size=16,
        learning_rate=0.00001,
        beta=0,  # normalization loss parameter
        gamma=0,  # m2 loss function

        fold=1,
        model='u',
        num_classes=4,

        print_freq=10,
        do_load_checkpoint=False,
        checkpoint_dir="",

        # datasets=['TA', 'lv_papillary'],
        datasets=['left_outflow', 'right_outflow', 'TA', 'lv_papillary'],
        input_channels=12,
        orig_dir=os.path.join(data_root_dir, "orig_data"),
        new_dir=os.path.join(data_root_dir, 'ecg_data'),
        data_dir=os.path.join(data_root_dir, 'preprocess'),
        result_dir=os.path.abspath("results"),
        split_dir=os.path.join(data_root_dir, 'preprocess')
    )

    print(c)
    return c

