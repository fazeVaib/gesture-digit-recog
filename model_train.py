import numpy as np # for matrix calculations
from sklearn.svm import SVC # SVM classifier
from tensorflow import keras # for importing dataset
from sklearn.externals import joblib  # for saving our model
from sklearn import preprocessing  # for standardizing the dataset
# Extract Histogram of Oriented Gradients (HOG) for a given image.
from skimage.feature import hog

# loading data from MNIST dataset in Keras in 2 parts, training and testing data
(traindata, trainlabel), (testdata, testlabel) = keras.datasets.mnist.load_data()

# checking the shapes of data
print(traindata.shape, trainlabel.shape, testdata.shape, testlabel.shape)

# Initializing the hog feature list
hog_feature_list = []
hog_test = []


# converting each training example into respestive hog values
for data in traindata:
    ft = hog(data, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), block_norm='L2')
    hog_feature_list.append(ft)


# converting each test example into respestive hog values
for data in testdata:
    ft = hog(data, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), block_norm='L2')
    hog_test.append(ft)

# converting hog features into array of float values
hog_features = np.array(hog_feature_list, 'float64')
hog_tests = np.array(hog_test, 'float64')

# confirming the shapes
print(hog_features, hog_features.shape)

# Preprocessing the data using StandardScaler 
# It normalizes the data, thus making it easier to carry out the calculations.
preproc = preprocessing.StandardScaler().fit(hog_features)

# Transforming both train and test data into preprocessed data.
hog_features = preproc.transform(hog_features)
hog_tests = preproc.transform(hog_tests)
print(hog_features, hog_features.shape)

# Making the classifier using SVM
clf = SVC(C=5, gamma=0.05)

# Fitting the training features into the classifier and checking the score.
clf.fit(hog_features, trainlabel)

# Score of the training examples
print(clf.score(hog_features, trainlabel))

# Saving the model (classifier along with preprocessed data) using the dump()
joblib.dump((clf, preproc), "mnist_clf_svm.pkl", compress=3)
#  Higher value means more compression,
# but also slower read and write times. Using a value of 3 is often a good compromise

# Checking the result on the test data, thus checking the performance of the model on unseen data
print(clf.score(hog_tests, testlabel))
