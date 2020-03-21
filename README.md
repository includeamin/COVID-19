# covid19-detection
detection of Covid-19 from X-ray images

## Train
``` python
python3 TRAIN.py -d dataset/test -m model_name.hdf5
```
**NOTE** :
- Model with `model_name.hdf5` will save in SavedModel directory.
- `ep` variable in TRAIN.py file is epoch count. you can change it.

example:
```sh
python3 TRAIN.py -d dataset/test -m amin.hdf5
```
result:
![train result](/ReadmeImages/train_result.png)

## Test 
``` sh
python3 test_model_10_images.py -d dataset/validation -m model_name.hdf5
```
load and show 10 labeled samples.
example:
![train result](/ReadmeImages/example_1.png)

 