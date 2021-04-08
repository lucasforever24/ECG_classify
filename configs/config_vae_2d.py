from trixi.util import Config
import os


def get_config():
    data_root_dir = os.path.abspath("data")

    c = Config(
        n_epochs=50,
        batch_size=64,
        learning_rate=0.00001,
        image_size=32,
        beta=4,  # loss normalization parameter
        loss_type='H',

        fold=1,

        print_freq=10,
        do_load_checkpoint=False,
        checkpoint_dir="",

        data_dir=os.path.join(data_root_dir, 'mnist_data'),
        result_dir=os.path.abspath("results"),
    )

    print(c)
    return c

