''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
import uuid
import os
import pandas as pd
import performance_plots

import get_images
import get_landmarks

from sklearn.multiclass import OneVsRestClassifier as ORC
import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.svm import SVC as svm
from sklearn.neural_network import MLPClassifier as mlp


#identifies image directory and loads images
image_directory = '../f22-train-dataset'
X_train, y_train = get_images.get_images(image_directory)

#sets landmarks for images
X_train, y_train = get_landmarks.get_landmarks(X_train, y_train, 'landmarks/', 68, False)

test_images_location = '../f22-webcam-Nathan-dataset/47'
#test_images_location = '../f22-webcam-Courtney-dataset/48'

key = cv2. waitKey(1)
webcam = cv2.VideoCapture(0)

while True:
    try:
        check, frame = webcam.read()
        #print(check) #prints true as long as the webcam is running
        #print(frame) #prints matrix values of each framecd 
        cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        if key == ord('s'): 
            filename = str(uuid.uuid4()) + '.png'
            cv2.imwrite(os.path.join(test_images_location, filename), img=frame)
            webcam.release()
            
            #saves image            
            print("Processing image...")
            print("Image saved!")
            
            #cv2.destroyAllWindows()
            webcam = cv2.VideoCapture(0)
            continue
        elif key == ord('q'):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
        
    except(KeyboardInterrupt):
        print("Turning off camera.")
        webcam.release()
        print("Camera off.")
        print("Program ended.")
        cv2.destroyAllWindows()
        break

test_images = '../f22-webcam-Nathan-dataset'
#test_images = '../f22-webcam-Courtney-dataset'

X_test, y_test = get_images.get_images(test_images)
X_test, y_test = get_landmarks.get_landmarks(X_test, y_test, 'landmarks/', 68, False)

#Matching and Decision - Classifer 1 
clf = ORC(knn()).fit(X_train, y_train)
matching_scores_knn = clf.predict_proba(X_test)

#Matching and Decision - Classifer 2
clf = ORC(mlp()).fit(X_train, y_train)
matching_scores_svm = clf.predict_proba(X_test)

# Fuse scores
matching_scores = (matching_scores_knn + matching_scores_svm) / 2.0

gen_scores = []
imp_scores = []
classes = clf.classes_
matching_scores = pd.DataFrame(matching_scores, columns=classes)

for i in range(len(y_test)):    
    scores = matching_scores.loc[i]
    mask = scores.index.isin([y_test[i]])
    gen_scores.extend(scores[mask])
    imp_scores.extend(scores[~mask])
    
performance_plots.performance(gen_scores, imp_scores, 'kNN-MLP-score_fusion', 100)
