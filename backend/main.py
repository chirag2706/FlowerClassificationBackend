
from flask import Flask, request,redirect,jsonify
from flask_restful import Resource, Api
import tensorflow as tf
import numpy as np
import cv2


app = Flask(__name__)
api = Api(app)

CATEGORIES = ['bauhinia_blakeana','frangipani','ixora','jessamine','red_frangipani','white_frangipani']
model = tf.keras.models.load_model('https://drive.google.com/drive/folders/1-N8aPPJwIk89GD9po_WE2o7m7Ksj8qHK?usp=sharing')
class FlowerClassifierPredictionQuery(Resource):

    ht = 128
    wd = 128

    def center(self,img):
        img = np.array(img)  
        img = cv2.resize(img, (self.wd, self.ht))

        img = img/255.0
        img[:,:,0] -= 0.5519912
        img[:,:,1] -= 0.4811025
        img[:,:,2] -= 0.4498843
        # print(img.shape)
        return img


    def get(self,encodedImage):
        img = tf.keras.preprocessing.image.load_img(encodedImage)
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = self.center(img)
        pred = model.predict(tf.convert_to_tensor([img]))
        print(CATEGORIES[np.argmax(pred[0])])
        return CATEGORIES[np.argmax(pred[0])]

    

#The below code tells that which calls should be called based on given URL.
api.add_resource(FlowerClassifierPredictionQuery,'/flower_classifier_prediction/<encodedImage>')


if __name__ == '__main__':
    #app is running of port 6615
    app.run(port='6615')