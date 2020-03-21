from bson import ObjectId
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, validator
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
import cv2
from pymongo import MongoClient
import os

app = FastAPI()
ClassLabels = ["covid", "normal", "vira neumonia"]

uploads_collection = MongoClient(host=os.environ["DATABASE_URL"]).get_database("COVID19").get_collection("uploads")


class UploadCollectionDataModel(BaseModel):
    file_name: str
    system_predict: str
    user_recommend: str

    @validator("system_predict")
    def system_predict_validator(cls, v):
        if v not in ClassLabels:
            raise HTTPException(detail="invalid label", status_code=400)
        return v


class UpdateUserRecommend(BaseModel):
    file_id: str
    user_recommend: str

    @validator("user_recommend")
    def system_predict_validator(cls, v):
        if v not in ClassLabels:
            raise HTTPException(detail=f"valid label is {ClassLabels} ", status_code=400)
        return v

    @validator("file_id")
    def validate_file_id(cls, v):
        if not ObjectId.is_valid(v):
            raise HTTPException(detail="invalid file id", status_code=400)
        return v


class DataBase:
    @staticmethod
    def add_new_uploaded_file(data: UploadCollectionDataModel):
        item = uploads_collection.insert_one(data.dict())
        return str(item.inserted_id)

    @staticmethod
    def add_user_predict_to_file(data: UpdateUserRecommend):
        exist = uploads_collection.find_one({"_id": ObjectId(data.file_id)})
        if exist['system_predict'] == data.user_recommend:
            same = True
        else:
            same = False
        uploads_collection.update_one({"_id": ObjectId(data.file_id)}, {
            "$set": {
                "user_recommend": data.user_recommend,
                "predict_true": same
            }
        })


class LabelImage:
    Model_Path = './SavedModel/amin.hdf5'
    UploadFolder = './Files/'

    @staticmethod
    def path_creator(image_name: str) -> str:
        return f"{LabelImage.UploadFolder}{image_name}"

    @staticmethod
    def predict_file_name_creator(image_name: str):
        return f"{LabelImage.UploadFolder}predict_{image_name}"

    @staticmethod
    async def label_the_image(image_name: str):
        image_path = LabelImage.path_creator(image_name)
        image = cv2.imread(image_path)
        size = 50
        sp = SimplePreprocessor(size, size)
        iap = ImageToArrayPreprocessor()
        sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
        (data, labels) = sdl.single_load(image_path)
        data = data.astype("float") / 255.0
        model = load_model(LabelImage.Model_Path)
        predict = model.predict(data, batch_size=size).argmax(axis=1)[0]
        cv2.putText(image, "Label: {}".format(ClassLabels[predict]),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imwrite(LabelImage.predict_file_name_creator(image_name), image_path)


@app.post("/upload/x-ray")
async def create_upload_file(b_file: bytes = File(...), file: UploadFile = File(...)):
    if len(b_file) > 4e+6:
        raise HTTPException(detail="too large, file should be max 5 MB", status_code=405)
    return {"filename": file.filename}
