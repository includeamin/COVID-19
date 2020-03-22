from tempfile import NamedTemporaryFile
from typing import IO

from bson import ObjectId
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, BackgroundTasks
from pydantic import BaseModel, validator
from keras.models import load_model
from Utils.ImageTools import ImageToArrayPreprocessor
from PrePorcessor.Preprocessor import SimplePreprocessor
from dataset.SimpleDatasetLoader import SimpleDatasetLoader
import cv2
from pymongo import MongoClient
import os
from uvicorn import run
from starlette import status
import shutil
from datetime import datetime
import keras
from fastapi.requests import Request
from fastapi.responses import FileResponse

print(keras.__version__)
app = FastAPI()
from fastapi.templating import Jinja2Templates

ClassLabels = ["covid", "normal", "vira neumonia"]

uploads_collection = MongoClient(host=os.environ["DATABASE_URL"]).get_database("COVID19").get_collection("uploads")


class UploadCollectionDataModel(BaseModel):
    file_name: str
    system_predict: str
    user_recommend: str = None
    create_at: datetime = datetime.now()

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
    Model_Path = './SavedModel/model_keras_215.hdf5'
    UploadFolder = './Files/'

    @staticmethod
    def path_creator(image_name: str) -> str:
        return f"{LabelImage.UploadFolder}{image_name}"

    @staticmethod
    async def label_the_image(image_path: str, file_name: str, user_recommendation: str):
        # print(image_name)
        # image_path = LabelImage.path_creator(image_name)
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
        cv2.imwrite(Tools.get_predicted_file_name(file_name), image)
        file_id = DataBase.add_new_uploaded_file(
            UploadCollectionDataModel(file_name=file_name, system_predict=ClassLabels[predict],
                                      user_recommend=user_recommendation))
        return ClassLabels[predict], file_id


class Tools:
    @staticmethod
    def get_secure_file_name(file_name: str):
        return f"{datetime.now().timestamp()}.{file_name.split('.')[-1]}"

    @staticmethod
    def get_predicted_file_name(file_name: str):
        return f"./Files/predicted_{file_name}"


async def valid_content_length(content_length: int = Header(..., lt=80_000)):
    return content_length


@app.post("/upload/x-ray")
async def create_upload_file(
        file: UploadFile = File(...),
        user_recommendation: str = None
):
    real_file_size = 0
    temp: IO = NamedTemporaryFile(delete=False)
    secure_file_name = Tools.get_secure_file_name(file.filename)
    for chunk in file.file:
        real_file_size += len(chunk)
        # if real_file_size > file_size:
        #     raise HTTPException(
        #         status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Too large"
        #     )
        temp.write(chunk)
    temp.close()
    image_path = f"./Files/{secure_file_name}"
    shutil.move(temp.name, image_path)
    if file.file.__sizeof__() > 4e+6:
        raise HTTPException(detail="too large, file should be max 5 MB", status_code=405)
    result, file_id = await LabelImage.label_the_image(image_path, secure_file_name, user_recommendation)
    return {"predict": result, "file_id": file_id}


templates = Jinja2Templates(directory="templates")


@app.get("/")
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/dist/dropzone.js")
def get_dpks():
    return FileResponse("Templates/dropzone.js")


if __name__ == '__main__':
    run(app)
