import numpy as np
from sklearn.svm import SVC
from tensorflow import keras
from sklearn.externals import joblib  # for saving our model
from sklearn import preprocessing  # for standardizing the dataset
# Extract Histogram of Oriented Gradients (HOG) for a given image.
from skimage.feature import hog

(traindata, trainlabel), (testdata, testlabel) = keras.datasets.mnist.load_data()
print(traindata.shape, trainlabel.shape, testdata.shape, testlabel.shape)

hog_feature_list = []
hog_test = []

for data in traindata:
    ft = hog(data, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), block_norm='L2')
    hog_feature_list.append(ft)

for data in testdata:
    ft = hog(data, orientations=9, pixels_per_cell=(
        14, 14), cells_per_block=(1, 1), block_norm='L2')
    hog_test.append(ft)

hog_features = np.array(hog_feature_list, 'float64')
hog_tests = np.array(hog_test, 'float64')

print(hog_features, hog_features.shape)

preproc = preprocessing.StandardScaler().fit(hog_features)
hog_features = preproc.transform(hog_features)
hog_tests = preproc.transform(hog_tests)
print(hog_features, hog_features.shape)

clf = SVC(C=5, gamma=0.05)
clf.fit(hog_features, trainlabel)

print(clf.score(hog_features, trainlabel))

joblib.dump((clf, preproc), "mnist_clf_svm.pkl", compress=3)
#  Higher value means more compression,
# but also slower read and write times. Using a value of 3 is often a good compromise

print(clf.score(hog_tests, testlabel))
