# facemask-detection-heroku

## MODEL
* This is a single shot detector(SSD) with backbone as MobileNetV3Large. SSD takes 3 features from backbone and 5 extra layers with aspect ratio [2,3].
* Checkout the detectionUtils.py

## API
* base url: https://facemask-detection-api.herokuapp.com/
* endpoint: predict
* request params: (@formdata) filename:imagefile
* request type: POST

![Screenshot from 2021-10-16 00-38-53](https://user-images.githubusercontent.com/27999714/137540182-e511e4ce-fa36-418e-a802-1b5058ff733f.png)



