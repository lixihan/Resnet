import io
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from tensorflow import keras
import tensorflow as tf


app = Flask(__name__)

model = None


def load_model():
    # load the pre-trained Keras model
    global model
    model = keras.models.load_model('res.h5')
    global graph
    graph = tf.get_default_graph()

def prepare_image(image, target):
    # if the image mode is not RGB, convert it
    if image.mode != "RGB":
        image = image.convert("RGB")

    # resize the input image and preprocess it
    image = image.resize(target)
    image = keras.preprocessing.image.img_to_array(image)
    image = np.expand_dims(image, 0)
    image = keras.applications.resnet50.preprocess_input(image)

    # return the processed image
    return image

@app.route("/predict", methods=["POST"])
def predict():
    # initialize the data dictionary that will be returned from the
    # view
    data = {"success": False}

    # ensure an image was properly uploaded to our endpoint
    if request.method == "POST":
        if request.files.get("image"):
            # read the image in PIL format
            image = request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            # preprocess the image and prepare it for classification
            image = prepare_image(image, target=(64, 64))

            # classify the input image and then initialize the list
            # of predictions to return to the client
            with graph.as_default():
                preds = model.predict(image)
                results = keras.applications.resnet50.decode_predictions(preds)
            data["predictions"] = []
            
            for (imagenetID, label, prob) in results[0]:
                r = {"label": label, "probability": float(prob)}
                data["predictions"].append(r)
      
            # indicate that the request was a success
            data["success"] = True

    # return the data dictionary as a JSON response
    return jsonify(data)

# if this is the main thread of execution first load the model and
# then start the server
if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()
