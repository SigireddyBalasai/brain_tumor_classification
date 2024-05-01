from flask import Flask, request
import tensorflow as tf

app = Flask(__name__)
print(tf.saved_model.contains_saved_model("/workspaces/brain_tumor_classification/model/model"))

# Define the load options
load_options = tf.saved_model.LoadOptions()


# Load the model
model = tf.saved_model.load('/workspaces/brain_tumor_classification/model/model', options=load_options)

@app.route('/inference', methods=['POST'])
def inference():
    # Get the image file from the request
    image_file = request.files['image']

    # Perform inference on the image
    image = tf.keras.preprocessing.image.load_img(image_file, target_size=(224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    prediction = model(image)
    predicted_class = tf.keras.applications.mobilenet.decode_predictions(prediction.numpy(), top=1)[0][0][1]

    return predicted_class

if __name__ == '__main__':
    app.run()
