# covid19-detection
Detection of Covid-19 from X-ray images

## Status:

![alter](https://includeamin.com/covid/github/stats/total_uploaded.svg)
![alter](https://includeamin.com/covid/github/stats/total_correct_predict.svg)
![alter](https://includeamin.com/covid/github/stats/total_wrong_predict.svg)
![alter](https://includeamin.com/covid/github/stats/test_accuracy.svg)
![alter](https://includeamin.com/covid/github/stats/train_accuracy.svg)
![alter](https://includeamin.com/covid/github/stats/model_in_use.svg)
[![Donate](https://img.shields.io/badge/Donate-PayPal-green.svg)](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=FEFTEUJT3YPDJ)


## Train
``` shell script
python3 TRAIN.py -d dataset/test -m model_name.hdf5
```
**NOTE** :
- Model with `model_name.hdf5` will save in SavedModel directory.
- `ep` variable in TRAIN.py file is epoch count. you can change it.

example:
```shell script
python3 TRAIN.py -d dataset/test -m amin.hdf5
```
result:
![train result](/ReadmeImages/train_result.png)

## Test 
### test 10 random image
``` shell script
python3 test_model_10_images.py -d dataset/validation -m model_name.hdf5
```
load and show 10 labeled samples

example:

![test result](/ReadmeImages/example_1.png)

### test single image

```shell script
python3 test_model_1_image.py -i path_to_image  -m path_to_model

```
example:
```shell script
python3 test_model_1_image.py -i ./dataset/one/covid/Chest.jpeg  -m ./SavedModel/amin.hdf5
```

## API ( WIP )
for gathering more images and make the dataset better, I create e simple API for upload the X-RAY image like below examples:
The uploaded image will validate after upload and the server return label of the image.
Label my be incorrect. (because of the low count of images in the dataset).

![](/dataset/validation/covid/01.jpeg)
![](/dataset/validation/covid/02.jpeg)
![](/dataset/validation/covid/03.jpeg)

## API Docs
### run on your host
- add mongodb address to `conf.env` file
- run `sudo docker-compose up -d`
- open docs `http://host:8838/docs`

## Website Version
- Work In Progress

## Flutter Version
- Flutter Version will develope by @EhsanTgv
