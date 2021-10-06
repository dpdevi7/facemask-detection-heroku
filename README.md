# facemask-detection-heroku

* Flask app deployed on heroku.
* This is a single shot detector(SSD) with backbone as MobileNetV3Large. SSD takes 3 features from backbone and 5 extra layers with aspect ratio [2,3].

# API
* base url: https://facemask-detection-api.herokuapp.com/
* endpoint: predict
* request params: (@formdata) filename:imagefile
* request type: POST
* response: returns json object of 3 keys: boxes, labels, scores



