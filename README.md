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


![Screenshot from 2021-10-07 00-26-11](https://user-images.githubusercontent.com/27999714/136266233-d28e9198-d1d7-456b-a937-aa5ed4424833.png)

