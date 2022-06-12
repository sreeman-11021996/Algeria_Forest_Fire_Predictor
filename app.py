<<<<<<< HEAD
import pickle
from flask import Flask, request, app, jsonify, url_for, render_template
import pandas as pd
import numpy as np
import trf

app = Flask(__name__)
model = pickle.load(open("model_new_aff.pkl", "rb"))
trf_model = pickle.load(open("X_trf_model.pkl", "rb"))

# Get your X_train
mongo = trf.mongodb()
X = mongo.get_collection()
# fit your X_train
trf.transform(X,trf_model)wq

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/batch')
def batch():
    return render_template("upload.html")

@app.route('/predict_post', methods=['POST'])
def predict_post():
    data = request.json['data']
    new_data = [list(data.values()),]
    print(new_data)
    print(trf.transform.X_trf_transform(new_data))

    output = model.predict(new_data)[0]
    return jsonify(output)


@app.route('/predict_html', methods=['POST'])
def predict_html():
    data = [float(x) for x in request.form.values()]
    print(data)
    final_features = [np.array(data)]
    print(data)
    print(trf.transform.X_trf_transform(data))

    output = model.predict(final_features)[0]
    if output:
        result = "not fire"
    else:
        result = "fire"
    #print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html',prediction_text="Algerian forest -> {}"
                           .format(result))

@app.route("/predict_html_batch")
def predict_html_batch():
    file = request.files["myfile"]

    df = pd.read_csv(file)
    pass


if __name__=="__main__":
    app.run(debug=True)

=======
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
    #print(data)

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

>>>>>>> 711a196be9297c4bc3ae716d5b9b5ffc417514d4
