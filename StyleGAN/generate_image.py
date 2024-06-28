from stylegan_utils import generate_examples, load_all
from classes import Generator


def initialize_model(identifier):
    model = load_all(identifier)
    step, alpha, z_dim, w_dim, in_channels, img_channels = model['parameters']
    gen = Generator(z_dim=z_dim, w_dim=w_dim, in_channels=in_channels, img_channels=img_channels)
    gen.load_state_dict(model['generator'])
    return gen, step, alpha, z_dim

gen = initialize_model('first')
generate_examples(gen, step, z_dim, n=100, device='cpu')
