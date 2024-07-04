from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/")
def hello():
    return jsonify({'data': "Hello, World!"})


@app.route('/post', methods=["POST"])
def testpost():
    input_json = request.get_json(force=True)
    dictToReturn = {'text_return': input_json['text']}
    return jsonify(dictToReturn)


app.run(debug=True)
