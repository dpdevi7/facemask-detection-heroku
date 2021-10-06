# facemask-detection-heroku

## MODEL
* This is a single shot detector(SSD) with backbone as MobileNetV3Large. SSD takes 3 features from backbone and 5 extra layers with aspect ratio [2,3].
* Checkout the detectionUtils.py

## API
* base url: https://facemask-detection-api.herokuapp.com/
* endpoint: predict
* request params: (@formdata) filename:imagefile
* request type: POST
* response: returns json object of 3 keys: boxes, labels, scores


![Screenshot from 2021-10-07 00-31-56](https://user-images.githubusercontent.com/27999714/136266497-b65e8e5e-8d8a-48b1-837f-29f38dc6cb97.png)


