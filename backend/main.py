from flask import Flask, request, jsonify
from utils import load_generator, generate_examples
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
    # input_json = request.get_json(force=True)
    imgs = generate_examples(gen, 6, n=5)
    pil_imgs_64 = serve_pil_image64(imgs)
    dictToReturn = {'text_return': 'text', 'num': len(pil_imgs_64), 'img': pil_imgs_64}
    # dictToReturn = {'text_return': 'text', 'num': len(imgs)}
    print('Sending 5 images')
    return jsonify(dictToReturn)

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
    # return jsonify({'status': True, 'image': img_str})

if __name__ == "__main__":
    global gen
    gen = load_generator()

    app.run(debug=True)
