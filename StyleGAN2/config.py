import torch.cuda as cuda

DATASET = "/vol/bitbucket/ik323/fyp/goodboy"
# DATASET = "../data/dataset2"
DEVICE = "cuda" if cuda.is_available() else "cpu"
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 8
LOG_RESOLUTION = 8 #for 256*256
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10