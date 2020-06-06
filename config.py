NOISE_DIM = 100
IMAGE_DIM = (100, 72)

PROGRESS_BAR = {
    'width': 100,
    'positive': '+',
    'negative': '-'
}

BUFFER_SIZE = 100000
BATCH_SIZE = 256

data_file = './data/mtg_art.npy'
update_data_file = False

NUM_EXAMPLES_TO_GENERATE = 16

LEARNING_RATE = 1e-4
EPOCHS = 100

checkpoint_frequency = 20
checkpoint_restore = True
checkpoint_dir = './training_checkpoints'
use_tensorboard = False