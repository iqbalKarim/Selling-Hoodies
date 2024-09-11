from flask import Flask, request, jsonify
from utils import *
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms.v2 as transforms
from flask_cors import CORS
from stgan2helper import generate_images
from NST.helper import convert_image
import json
from torchvision.models import vgg19, VGG19_Weights

app = Flask(__name__)
CORS(app)

# @app.route("/")
# def hello():
#     return jsonify({'data': "Hello, World!"})


@app.route('/', methods=["GET"])
def testpost():
    normalize = request.args.get("normalize")
    imgs = generate_examples(gen, 6, n=5, normalize=normalize)
    pil_imgs_64 = serve_pil_image64(imgs)
    dictToReturn = {'text_return': 'text', 'num': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dictToReturn)

@app.route("/emnist", methods=["GET"])
def emnistGet():
    count = request.args.get('count')
    gridDim = request.args.get('gridDim')
    imgs = generate_grid(emnistGen, 3, cols=int(gridDim), n=int(count))
    pil_imgs_64 = serve_pil_image64(imgs)
    dict_return = {'count': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dict_return)

@app.route("/graffiti", methods=["GET"])
def graffitiGet():
    imgs = generate_examples(graffitiGen, steps=6, n=5)
    pil_imgs_64 = serve_pil_image64(imgs)
    dict_return = {'count': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dict_return)

@app.route("/mnist", methods=["GET"])
def mnistGet():
    count = request.args.get('count')
    gridDim = request.args.get('gridDim')
    imgs = generate_grid(mnistGen, 3, cols=int(gridDim), n=int(count))
    pil_imgs_64 = serve_pil_image64(imgs)
    dict_return = {'count': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dict_return)

@app.route("/metfaces", methods=["GET"])
def metfacesGet():
    count = request.args.get('count')
    imgs = generate_images(metFacesGen, outdir='out', truncation_psi=0.7, n=count)
    pil_imgs_64 = serve_pil_image64(imgs)
    dict_return = {'count': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dict_return)

@app.route("/nst", methods=['POST'])
def nstApply():
    body = request.json
    # print(body)
    im_bytes = base64.b64decode(body["content"])  # im_bytes is a binary image
    im_file = BytesIO(im_bytes)  # convert image to file-like object
    img = Image.open(im_file)
    # img.show()

    path_style = body["style"]["src"]
    img_style = Image.open(f'./nstStyles/{path_style}')
    # img_style.show()

    result_img = convert_image(img_style, img, cnn_normalization_std, cnn_normalization_mean, cnn, num_steps=100)
    img_io = BytesIO()
    result_img.save(img_io, 'PNG', quality=100)
    img_str = base64.b64encode(img_io.getvalue()).decode("utf-8")

    return jsonify({'image': img_str})

def serve_pil_image64(pil_img):
    # pil_img = transforms.ToPILImage()(imgs[0][0]).convert("RGB")
    transformer = transforms.ToPILImage()
    imgs = []
    for img in pil_img:
        img_io = BytesIO()
        img2 = transformer(img)
        img2.save(img_io, 'PNG', quality=100)
        img_str = base64.b64encode(img_io.getvalue()).decode("utf-8")
        imgs.append(img_str)

    return imgs
    # return img_str


if __name__ == "__main__":
    global gen
    global emnistGen
    global graffitiGen
    global mnistGen
    global metFacesGen
    global nstVgg
    global cnn_normalization_mean
    global cnn_normalization_std
    global cnn

    cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()
    print('NST CNN loaded')

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])

    gen = load_generator()
    emnistGen = load_emnist_generator()
    graffitiGen = load_graffiti_generator()
    mnistGen = load_mnist_generator()
    metFacesGen = load_met()

    app.run(debug=True)
