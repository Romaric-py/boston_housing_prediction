import inference
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

@app.route("/",  methods=["GET", "POST"])
def index():
    return render_template('index.html', variables = inference.boston_variables)

@app.route("/predict", methods=["POST"])
def predict():
    response = jsonify(inference.predict(request.get_json()))
    return response


if __name__ == '__main__':
    app.run(port=8000, debug=True)