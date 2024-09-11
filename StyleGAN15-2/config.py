import torch.cuda as cuda

DATASET = "/vol/bitbucket/ik323/fyp/dataset"
# DATASET = "/vol/bitbucket/ik323/fyp/graffiti"
# DATASET = "../data/dataset2"
DEVICE = "cuda" if cuda.is_available() else "cpu"
EPOCHS = 300
STEP_EPOCHS = 40
LEARNING_RATE = 1e-3
BATCH_SIZE = 4
BATCH_SIZES = [512, 256, 128, 64, 64, 64, 32]
# BATCH_SIZES = [64, 8, 8, 8, 8, 8, 8]
# PROGRESSIVE_EPOCHS = [30] * len(BATCH_SIZES)
LOG_RESOLUTION = 8 #for 256*256
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10

START_TRAIN_AT_IMG_SIZE = 8 #The authors start from 8x8 images instead of 4x4
CHANNELS_IMG = 3
IN_CHANNELS = 256
factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]