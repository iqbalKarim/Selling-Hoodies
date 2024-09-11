if __name__ == "__main__":
    print('STYLEGAN 2')
    n_features = 8
    max_features = 256
    LOG_RESOLUTION = 8  # for 256*256
    Z_DIM = 256
    in_channels = 256
    W_DIM = 256
    factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

    features = [min(max_features, n_features * (2 ** i)) for i in range(LOG_RESOLUTION, -1, -1)]
    n_blocks = len(features)
    blocks = [(W_DIM, features[i - 1], features[i]) for i in range(1, n_blocks)]
    print('\t\tGenerator')
    print('\tfeatures: ', features)
    print('\tn_blocks: ', n_blocks)
    print('\tblocks: ', blocks)

    features = [int(min(max_features, n_features * (2 ** i))) for i in range(LOG_RESOLUTION + 1)]
    n_blocks = len(features) - 1
    blocks = [(features[i], features[i + 1]) for i in range(n_blocks)]
    print('\t\tDiscriminator')
    print('\tfeatures: ', features)
    print('\tn_blocks: ', n_blocks)
    print('\tblocks: ', blocks)


    print('\nSTYLEGAN 1')
    print('\t\tGenerator')
    gen_prog = [(W_DIM, int(in_channels * factors[i]), int(in_channels * factors[i+1])) for i in range(len(factors) - 1)]
    print('\t(w_dim, in, out): ', gen_prog)

    print('\t\tDiscriminator')
    disc_prog = [(int(in_channels * factors[i]), int(in_channels * factors[i - 1])) for i in range(len(factors) - 1, 0, -1)]
    print('\t(in, out): ', disc_prog)