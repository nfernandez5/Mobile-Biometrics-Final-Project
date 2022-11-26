''' Imports '''
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import get_images
import get_landmarks
import performance_plots

from sklearn.multiclass import OneVsRestClassifier as ORC
import pandas as pd

''' Import classifiers '''
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.naive_bayes import MultinomialNB as mnb
from sklearn.svm import SVC as svm
from sklearn.neural_network import MLPClassifier as mlp

''' Load training data and their labels '''
X_train, y_train = get_images.get_images('../f22-train-dataset') # when training on controlled set, we need to remove the images that we are testing with

''' Get distances between face landmarks in the images '''
X_train, y_train = get_landmarks.get_landmarks(X_train, y_train, 'landmarks/', 68, False)

'''Load test data and their labels'''
#X_test, y_test = get_images.get_images('../f22-aged-test-dataset')
X_test, y_test = get_images.get_images('../f22-control-dataset')

''' Get distances between face landmarks in the images '''
X_test, y_test = get_landmarks.get_landmarks(X_test, y_test, 'landmarks/', 68, False)

''' Matching and Decision - Classifer 1 '''
#change classifiers to see if ones are better than others
clf = ORC(svm(probability=(True))).fit(X_train, y_train)
matching_scores_knn = clf.predict_proba(X_test)

''' Matching and Decision - Classifer 2 '''
clf = ORC(mlp()).fit(X_train, y_train)
matching_scores_svm = clf.predict_proba(X_test)

''' Fuse scores '''
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
    
performance_plots.performance(gen_scores, imp_scores, 'SVM-MLP-control-score_fusion', 100)
