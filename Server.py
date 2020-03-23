from tempfile import NamedTemporaryFile
from typing import IO

from bson import ObjectId
from fastapi import FastAPI, UploadFile, File, HTTPException, Header, Depends, BackgroundTasks
from pybadges import badge
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
from fastapi.responses import Response, StreamingResponse

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
    is_correct: bool = False

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
        is_correct: bool = True if user_recommendation == ClassLabels[predict] else False
        file_id = DataBase.add_new_uploaded_file(
            UploadCollectionDataModel(file_name=file_name, system_predict=ClassLabels[predict],
                                      user_recommend=user_recommendation, is_correct=is_correct))
        return ClassLabels[predict], file_id


class Tools:
    @staticmethod
    def get_secure_file_name(file_name: str):
        return f"{datetime.now().timestamp()}.{file_name.split('.')[-1]}"

    @staticmethod
    def get_predicted_file_name(file_name: str):
        return f"./Files/predicted_{file_name}"

    @staticmethod
    def pagination(page=1):
        PAGE_SIZE = 15
        x = page - 1
        skip = PAGE_SIZE * x
        return skip, PAGE_SIZE

    @staticmethod
    def mongo_id_fix(data: dict):
        data["_id"] = str(data["_id"])
        return data


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


@app.get("/get_last_uploads_result")
def get_last_result():
    temp = uploads_collection.aggregate([
        {
            "$group": {
                "_id": "$_id",
                "total_correct_predict": {
                    "$sum": {"$cond": ["$is_correct", 1, 0]}
                },
                "total_wrong_predict": {
                    "$sum": {"$cond": ["$is_correct", 0, 1]}
                },
                "total_uploaded": {"$sum": 1}

            }
        }
    ])
    resp_2 = [item for item in temp][0]
    resp_2.pop("_id")
    resp_2["train_accuracy"] = "99%"
    resp_2["test_accuracy"] = '91%'
    resp_2["model_in_use"] = 'model_keras_215.hdf5'

    return {
        "stats": resp_2
    }


@app.get("/github/stats/{name}.svg")
def get_stats_icon(name: str):
    valid = ["total_correct_predict",
             'total_wrong_predict',
             "total_uploaded",
             "test_accuracy",
             "train_accuracy",
             "model_in_use"]
    if name not in valid:
        raise HTTPException(detail="not found", status_code=404)
    result = get_last_result()['stats'][name]
    s = badge(left_text=name.replace('_', ' '), right_text=str(result),
              right_color='green' if name != 'total_wrong_predict' else 'red')
    return Response(content=s, media_type='image/svg+xml', status_code=200)


if __name__ == '__main__':
    run(app)
