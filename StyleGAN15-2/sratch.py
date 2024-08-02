max_features = 256
n_features = 8
log_resolution = 8
W_DIM = 256

print('Discriminator')
disc_features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 1)]
disc_n_blocks = len(disc_features) - 1
disc_blocks = [(disc_features[i], disc_features[i + 1]) for i in range(disc_n_blocks)]
disc_final_features = disc_features[-1] + 1

print("\tFeatures: ", disc_features)
print(f"\tN blocks: {disc_n_blocks}")
print('\tblocks', *disc_blocks)
print(f"\tfinal features: {disc_final_features}")
disc_finals = []
for i in range(1, disc_n_blocks + 1):
    disc_finals.append(disc_features[i])

print(f"\tDisc Finals: ", disc_finals)


print('\n\nGenerator')
gen_features = [min(max_features, n_features * (2 ** i)) for i in range(log_resolution - 2, -1, -1)]
gen_n_blocks = len(gen_features)
print(f"\tInitial Constant features: {gen_features[0]}")
print(f"\tStyle Block: {(W_DIM, gen_features[0], gen_features[0])}")
print(f"\tToRGB Block: {(W_DIM, gen_features[0])}")

gen_blocks = [(W_DIM, gen_features[i - 1], gen_features[i]) for i in range(1, gen_n_blocks)]
print("\tfeatures: ", gen_features)
print("\tN blocks: ", gen_n_blocks)
print("\tblocks: ", gen_blocks)
gen_prelims = []
for i in range(0, gen_n_blocks-1):
    gen_prelims.append(gen_features[i])
print(f'\tGen Prelims: {gen_prelims}')
# print("\n\nDiscriminator")
# for step in range(1, disc_n_blocks + 1):
#     print("\tStep: ", step)
#     for i in range(step):
#         print(disc_blocks[i], end=" --> ")
#     _, final_features = disc_blocks[step - 1]
#     print(final_features + 1, "---", 1)
# print("Generator")
# for step in range(gen_n_blocks - 1, 0, -1):
#     print("\tStep: ", step)
#     print(gen_features[step-1], end=" -->-- ")
#     for i in range(step, gen_n_blocks):
#         print(gen_blocks[i - 1], end=" --> ")
#     print()

print()
for gen_step in range(gen_n_blocks - 1, 0, -1):
    disc_step = gen_n_blocks - gen_step
    # print(f"\t\tgen step: {gen_step} \tdisc step: {disc_step} ")
    print("Generator:", end="\t\t")
    print("noise res: ", log_resolution - gen_step + 1, end="\t\t")
    # print(gen_features[gen_step - 1], end=" -->-- ")
    print(gen_prelims[gen_step - 1], end=" -->-- ")
    for i in range(gen_step, gen_n_blocks):
        print(gen_blocks[i - 1], end=" --> ")
    print()
    # print("Discriminator:", end="\t")
    # for i in range(disc_step):
    #     print(disc_blocks[i], end=" --> ")
    # # _, final_features = disc_blocks[disc_step - 1]
    # final_features = disc_finals[disc_step - 1]
    # print(final_features + 1, "---", 1)
    #
    # print()

# for step in range(gen_n_blocks - 1, 0, -1):
#     critic_step = gen_n_blocks - step
#     print('step: ', critic_step)
#     for i in range(critic_step):
#         print(disc_blocks[i], end="--->")
#     print(disc_finals[critic_step - 1])
#
#
# for step in range(gen_n_blocks - 1, 0, -1):
#     size = min(256, 2 ** (gen_n_blocks - step + 2))
#     print(size)


