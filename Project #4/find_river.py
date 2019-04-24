from imutils import paths
import imutils
from sklearn.preprocessing import LabelEncoder
from skimage import feature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.externals import joblib
from sklearn.svm import LinearSVC
import numpy as np
import testing as test
import cv2

#### testing multiple block and threshold sizes

orient = 9
pix_per_cell = 8
cell_per_block = 1
neg_path = "/Users/amanda/Documents/School Work/Computer Vision/Project #4/Training/Neg"
train_path = "/Users/amanda/Documents/School Work/Computer Vision/Project #4/Training/River"
test_path= "/Users/amanda/Documents/School Work/Computer Vision/Project #4/Testing/1.jpg"

##### Block test 1

block = 25
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)



##### Block test 2

block = 100
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)

##### Block test 3

block = 125
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)

##### Block test 4

block = 40
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)


##### Block test 5

block = 60
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)


##### Block test 6

block = 15
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)

##### Block test 7

block = 150
labels=[]
data=[]

for imagePath in paths.list_images(train_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)
    gray = cv2.resize(gray, (block, block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True
                    )
    data.append(H)
    labels.append(1)

for imagePath in paths.list_images(neg_path):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edged = imutils.auto_canny(gray)

    gray = cv2.resize(gray, (block,block))

    H , hold = feature.hog(gray, orientations=orient,
                                      pixels_per_cell=(pix_per_cell, pix_per_cell),
                                      cells_per_block=(cell_per_block, cell_per_block),
                   block_norm='L2', feature_vector=True, visualise=True)
    data.append(H)
    labels.append(0)


data_array = np.array(data)
le = LabelEncoder()
labels = le.fit_transform(labels)
labels_array = np.array(labels)

print(data_array)

model = LinearSVC()
clf = CalibratedClassifierCV(model, method='sigmoid')
clf.fit(data_array, labels_array)
joblib.dump(clf, 'model.npy')
test.test(model,test_path,block,.10)
test.test(model,test_path,block,.20)
test.test(model,test_path,block,.30)
test.test(model,test_path,block,.40)
test.test(model,test_path,block,.50)
test.test(model,test_path,block,.05)
test.test(model,test_path,block,.33)
test.test(model,test_path,block,.35)
test.test(model,test_path,block,.37)
















