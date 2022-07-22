import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class catdogpredict:
    def __init__(self,filename):
        self.filename = filename

    def prediction(self):
        #Initial step loading the model
        model=load_model("cat_dog_model.h5")

        #Logging the summary
        with open('summary_report.txt','w') as fh:
        # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))

        #Finally, working on prediction
        imagedata=self.filename
        test_image=image.load_img(imagedata,target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)


        #Presenting the results in json format
        if result[0][0] == 1:
            prediction = 'dog'
            return [{ "image" : prediction}]
        else:
            prediction = 'cat'
            return [{ "image" : prediction}]

