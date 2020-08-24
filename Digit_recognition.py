import numpy as np
from cv2 import cv2
import tensorflow as tf
import requests
from Boxes_around_digits import extract_digits
import argparse
import imutils
from paper_extractor import four_point_transformation
import time

# create parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image")
ap.add_argument("-t", "--threshold", type=float, default=0.5)
ap.add_argument("-m", "--model", type=str, default="model.h5")
ap.add_argument("-u", "--url", type=str, default="http://192.168.0.100:8080/shot.jpg")
args = vars(ap.parse_args())


# Define function to preprocess image
def preProcessing(img_th):
    img_th = cv2.resize(img_th, (28, 28))
    img_th = img_th / 255
    img_th = img_th.reshape(1, 28, 28, 1)
    return img_th


# load model by path
def load_model(path):
    model = tf.keras.models.load_model(path)
    return model


# find objects and make predictions
def predict_objects(img, model, threshold):
    # Try to extract paper
    img = four_point_transformation(img)

    rects, digits = extract_digits(img)

    len_obj = digits.shape[0]

    if len_obj:
        for i in range(0, len_obj):
            obj = preProcessing(digits[i])

            # make predictions
            predictions = model.predict(obj)
            class_index = np.argmax(predictions)
            probval = np.amax(predictions)

            if probval > threshold:
                (startX, startY) = rects[i][0], rects[i][1]
                (endX, endY) = rects[i][0] + rects[i][2], rects[i][1] + rects[i][3]
                cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 10)
                cv2.putText(img, str(class_index), (startX - 5, startY - 5), cv2.FONT_HERSHEY_SIMPLEX, 3,
                            (0, 0, 255),
                            10)

    return img


if __name__ == "__main__":
    # Load model
    model = load_model(args['model'])

    # Define threshold
    threshold = args["threshold"]

    if args.get("image", False):
        img = cv2.imread(args["image"])

        img_pred = predict_objects(img, model, threshold)

        img_pred = imutils.resize(img_pred, width=500)

        cv2.imshow("AndroidCam", img_pred)
        cv2.waitKey(0)

    else:
        while True:
            try:
                img = requests.get(args["url"])
                img = np.array(bytearray(img.content), dtype=np.uint8)
                img = cv2.imdecode(img, - 1)
                img = cv2.resize(img, (600, 600))

                img_pred = predict_objects(img, model, threshold)

                cv2.imshow("AndroidCam", img_pred)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except requests.exceptions.ConnectionError:
                print("Connection error")
                break
