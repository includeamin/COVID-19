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


ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="halloo insert dataset")
ap.add_argument("-m", "--model", required=True, help="path to output model")
args = vars(ap.parse_args())
size = 50
classLabels = ["covid", "normal", "vira neumonia"]
print("[INFO] sampling images ...")
imagePaths = np.array(list(paths.list_images(args["dataset"])))
idxs = np.random.randint(0, len(imagePaths), size=(10,))
imagePaths = imagePaths[idxs]
sp = SimplePreprocessor(size, size)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths)
data = data.astype("float") / 255.0
print("[INFO] loading pre-trained network ...")
model = load_model(args["model"])
print("[INFO] predicting ...")
preds = model.predict(data, batch_size=size).argmax(axis=1)
print(preds)
for (i, imagePath) in enumerate(imagePaths):
    # load the example image, draw the prediction, and display it
    # to our screen
    image = cv2.imread(imagePath)
    # image=ReadyToUseImage(image)
    cv2.putText(image, "Label: {}".format(classLabels[preds[i]]),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
