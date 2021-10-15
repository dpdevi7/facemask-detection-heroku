from flask import Flask, request, jsonify, Response
from detectionUtils import getAllowedFileExtension
import detectionUtils
import os


app = Flask(__name__)
app.config["DEBUG"] = False # for deployment make it False

# 1 load model, always load model in global variable, so that every time a request hits, you dont need to
#   load model again and again.
MODEL = detectionUtils.loadModelPretrained()


@app.route('/predict', methods=['POST'])
def predict():

    responseObject = None
    statusCode = None
    print('filenames' in request.files)

    if request.method == 'POST':
        if 'filename' in request.files:
            imageFile = request.files['filename']
            imageFileName = imageFile.filename

            if getAllowedFileExtension(imageFileName):
                imgBytes = imageFile.read()
                imgTensor = detectionUtils.readImageNonScriptedModule(imgBytes)
                # 2 prediction
                boxes, labels, scores = detectionUtils.detectMaskNonScriptedModule(MODEL, imgTensor)
                # 3 if predictions are not empty
                # 4 return result
                responseObject = {
                    'success':True,
                    'result':{
                        'boxes': boxes,
                        'labels': labels,
                        'scores': scores
                    }
                }

                statusCode = 200
        else:
            responseObject = {
                'msg':"please send file with key -- filename",
                'success':False
            }
            statusCode = 400
    return jsonify(responseObject), statusCode

# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=8080)
# https://www.youtube.com/watch?v=bA7-DEtYCNM&t=528s
