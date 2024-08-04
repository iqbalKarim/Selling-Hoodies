from flask import Flask, request, jsonify
from utils import *
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms.v2 as transforms
from flask_cors import CORS

app = Flask(__name__)
CORS(app, origins="*")

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
    imgs = generate_grid(emnistGen, 3, cols=5)
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
    imgs = generate_grid(mnistGen, 3, n=5)
    pil_imgs_64 = serve_pil_image64(imgs)
    dict_return = {'count': len(pil_imgs_64), 'img': pil_imgs_64}
    return jsonify(dict_return)

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

    gen = load_generator()
    emnistGen = load_emnist_generator()
    graffitiGen = load_graffiti_generator()
    mnistGen = load_mnist_generator()

    app.run(debug=True)
