from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from keras.optimizers import SGD
from Model.IncludeNet import IncludeNet
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import cv2

classLabels = ["covid", "normal", "vira neumonia"]

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image_path", required=True, help="path to image")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())


def amin():
    image_path = args["image_path"]
    size = 50
    sp = SimplePreprocessor(size, size)
    iap = ImageToArrayPreprocessor()
    sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
    (data, labels) = sdl.single_load(image_path)
    data = data.astype("float") / 255.0
    model = load_model('./SavedModel/amin.hdf5')
    preds = model.predict(data, batch_size=size).argmax(axis=1)
    image = cv2.imread(image_path)
    # image=ReadyToUseImage(image)
    cv2.putText(image, "Label: {}".format(classLabels[preds[preds[0]]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    amin()
