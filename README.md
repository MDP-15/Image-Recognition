# MDP Image Recognition

This repository is forked from https://github.com/ultralytics/yolov5
### 1. Install dependencies
```
python3 -m pip install -r requirements.txt
```
### 2. Usage
Details of the `detect()` function in `dectect.py` is described as follows:

The function parameters we should take of note of are:
* `weights`: This holds the path of the file that stores the trained weights of the model. Default value is already set to be the path of our model (`mdp/weights/weights.pt`), so it does not need to be modified.
* `source`: This is the path of the input image file. Its value should be specifically specified.
* `output`: This is the path which holds the detection output image and text file. By default it outputs to `mdp/output`.
* `conf_thres`：This value specifies the threshold confidence level in order for a detection to be made. Default is `0.7`.

The function returns the predicted label of the input image as a string.