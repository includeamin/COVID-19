from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import cv2
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
from imutils import paths
import cv2
import numpy as np

app = FastAPI()


class LabelImage:
    @staticmethod
    def path_creator(image_name: str) -> str:
        return f"./UploadedFiles/{image_name}"

    @staticmethod
    async def label_the_image(image_name: str):
        image_path = LabelImage.path_creator(image_name)
        size = 50
        sp = SimplePreprocessor(size, size)
        iap = ImageToArrayPreprocessor()
        sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
        (data, labels) = sdl.single_load(image_path)
        data = data.astype("float") / 255.0
        model = load_model('./SavedModel/amin.hdf5')
        preds = model.predict(data, batch_size=size).argmax(axis=1)


@app.post("/upload/x-ray")
async def create_upload_file(b_file: bytes = File(...), file: UploadFile = File(...)):
    if len(b_file) > 4e+6:
        raise HTTPException(detail="too large, file should be max 5 MB", status_code=405)

    return {"filename": file.filename}
