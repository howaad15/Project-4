from skimage.feature import hog
from skimage.transform import pyramid_gaussian
from skimage import color
from imutils.object_detection import non_max_suppression
from sklearn.externals import joblib
import numpy as np
from sklearn.metrics import brier_score_loss
import cv2


orient = 9
pix_per_cell = 8
cell_per_block = 1



def sliding_window(image, stepSize, windowSize):
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            yield (x, y, image[y: y + windowSize[1], x:x + windowSize[0]])

def test(model, test_path, block, threshold):
    print("TESTINGGGG   thrs: "+str(threshold)) #just used for making sure the program was moving along
    scale = 0
    detections = []
    i=0
    model = joblib.load('/Users/amanda/PycharmProjects/Project #4/model.npy')
    img = cv2.imread(test_path)

    (winW, winH) = (block, block)
    windowSize = (winW, winH)
    downscale = 1.5

    for resized in pyramid_gaussian(img, downscale=1.5):
        for (x, y, window) in sliding_window(resized, stepSize=10, windowSize=(winW, winH)):

            #error testing
            if window.shape[0] != winH or window.shape[1] != winW:  # ensure the sliding window has met the minimum size requirement
                continue
            window = color.rgb2gray(window)
            fds = hog(window, orient, (pix_per_cell, pix_per_cell), (cell_per_block, cell_per_block),
                      block_norm='L2')  # extract HOG features from the window captured
            fds = fds.reshape(1, -1)

            pred=model.predict(fds)
            clf_score= brier_score_loss(y_true=[1],
                                        y_prob=model.predict_proba(fds)[:,1])

            if pred==[1]:

                if clf_score<threshold:
                    # create a list of all the predictions found
                    detections.append((int(x * (downscale ** scale)), int(y * (downscale ** scale)), clf_score,
                                       int(windowSize[0] * (downscale ** scale)),
                                       int(windowSize[1] * (downscale ** scale))))

        scale += 1

    rects = np.array([[x, y, x + w, y + h] for (x, y, _, w, h) in detections])  # do nms on the detected bounding boxes

    sc=[]
    for d in detections:
        sc.append(d[2])

    print("detection confidence score: ", sc)
    sc = np.array(sc)
    pick = non_max_suppression(rects, probs=sc, overlapThresh=0.3)

    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(img, (xA, yA), (xB, yB), (0, 255, 0), 2)
    #cv2.imshow("Raw Detections after NMS", img)

    s="block-"+str(block)+"__thresh-"+str(threshold)+".jpeg"
    cv2.imwrite(s, img)
    #cv2.waitKey(0)


