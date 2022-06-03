import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd
import sklearn

app = Flask(__name__)
model = pickle.load(open('model_aff.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_post', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    new_data = [list(data.values())]
    output = model.predict(new_data)[0]
    return jsonify(output)


@app.route('/predict_html', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)

    output = model.predict(final_features)[0]
    if output:
        result = "not fire"
    else:
        result = "fire"
    #print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html',
                           prediction_text="Algerian forest -> {}".format(
                               result))


if __name__=="__main__":
    app.run(debug=True)

