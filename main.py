from flask import Flask,jsonify,render_template,request
from flask_cors import CORS, cross_origin
import os
from utils.decoder import decodeImage
from predict import catdogpredict

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

#Initializing the application
app=Flask(__name__)
CORS(app)

class clientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = catdogpredict(self.filename)

    
@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('index.html')
    


@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    image = request.json['image']
    decodeImage(image, clApp.filename)
    result = clApp.classifier.prediction()
    return jsonify(result)


#port = int(os.getenv("PORT"))
if __name__ == "__main__":
    clApp = clientApp()
    #app.run(host='0.0.0.0', port=port)
    app.run(host='127.0.0.1', port=8000, debug=True)