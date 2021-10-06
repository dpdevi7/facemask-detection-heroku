from flask import Flask, request, jsonify
from detectionUtils import getAllowedFileExtension
import detectionUtils
import os


app = Flask(__name__)
app.config["DEBUG"] = True


@app.route('/predict', methods=['POST'])
def predict():

    responseObject = None

    if request.method == 'POST':
        imageFile = request.files['filename']
        imageFileName = imageFile.filename

        if getAllowedFileExtension(imageFileName):
            imgBytes = imageFile.read()
            imgTensor = detectionUtils.readImage(imgBytes)
            # 1 load model

            MODEL = detectionUtils.get_ssd_from_checkpoint(modelFile=os.path.join('model-dir','ssd.pth'),
                                                           SSD_MODEL=detectionUtils.get_ssd_model(4, 512))
            # 2 prediction
            boxes, labels, scores = detectionUtils.detectMask(MODEL, imgTensor)
            # 3 return result
            responseObject = {
                'boxes':boxes,
                'labels':labels,
                'scores':scores
            }




    return jsonify(responseObject)


if __name__ == "__main__":
    # app.run(host='127.0.0.1', port=5000)
    app.run(host='0.0.0.0', port=5000)