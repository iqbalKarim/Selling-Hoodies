import torch.cuda as cuda

# DATASET = "/vol/bitbucket/ik323/fyp/dataset"
DATASET = "../data/dataset"
DEVICE = "cuda" if cuda.is_available() else "cpu"
EPOCHS = 300
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
LOG_RESOLUTION = 3 #for 256*256
Z_DIM = 256
W_DIM = 256
LAMBDA_GP = 10