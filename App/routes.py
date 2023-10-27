from App import app, phase1_model
import App.utils.load as load
from App.utils.predict import predict_model
from flask import jsonify, request

@app.route('/', methods=['GET'])
def home():
    response = {'response': 'success'}
    return jsonify(response, 200)


@app.route('/upload', methods=['POST'])
def upload():
    # request.form[]
    # at the end of this function, the uploaded image and text label file (renamed to test.txt) should be uploaded into received/images and received/labels/test.txt respectively
    pass

@app.route('/predict', methods=['GET'])
def predict():
    image_folder = "App\\received\\images\\"
    label_filename = "App\\received\\labels\\test.txt"

    processed_data = load.preprocess(label_filename, image_folder, (128,128))
    results = predict_model(phase1_model, processed_data['x'])

    print(results)

    return results



