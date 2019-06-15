# gesture-digit-recog

Recognize the digits in real-time through shapes made in front of camera

## Contents

* "model_train.py" consists of the model trained using MNIST dataset implemented using SVM.
* "mnist_clf_svm.pkl" file consists of the saved model, which is later being loaded into the prediction function for preeiction.
* "camera_pred.py" is the file that carries out the prediction by accessing the camera. It displays the result in the original frame itself.

## Requirements

* openCV
* Numpy
* Scikit-image
* Sklearn
* Tensorflow
* Keras

## Operation

* The model is pre-trained and saved, but you can also train it with your own data.<br> 
* Run file 'camera_pred.py' in terminal.
<br>
* Use a green object to draw the digits in front of camera.
<br>
* press 'C' to clear the input and 'Q' to quit the program. 
