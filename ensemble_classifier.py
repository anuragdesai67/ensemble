# -*- coding: utf-8 -*-
"""
Created on Sun Apr  1 19:06:08 2018

@author: anurag
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB




def read_data(train_file,test_file):
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)
    x_train = train.iloc[:,0:4].values
    y_train = train.iloc[:,4].values
    x_test = test.iloc[:,0:4].values
    y_test = test.iloc[:,4].values
    return x_train,y_train,x_test,y_test



#function for random forest classifier
def classifier_random_forest(num_trees, seed, x_train, y_train, x_test,y_test):
    randomForest = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
    randomForest = randomForest.fit(x_train,y_train)
    y_pred = randomForest.predict(x_test)
    print("Number of trees: %d Accuracy: %0.4f" %(num_trees, (randomForest.score(x_test, y_test))))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    return randomForest
 
#function for AdaBoost classifier    
def classifier_adaboost(num_estimators, learning_rate,seed, x_train, y_train, x_test, y_test, decision_stump, ds_num):
    adaBoost = AdaBoostClassifier(base_estimator=decision_stump,learning_rate=learning_rate, n_estimators=num_estimators, random_state=seed, algorithm="SAMME")
    adaBoost = adaBoost.fit(x_train, y_train)
    y_pred = adaBoost.predict(x_test)
    print("Decision stump number:%d, number of estimators %d: Accuracy %0.4f" %(ds_num,num_estimators,(adaBoost.score(x_test, y_test))))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")
    return adaBoost
    

############ input files ###############
x_train,y_train,x_test,y_test = read_data("lab3-train.csv","lab3-test.csv")
seed = 200    

########### RandomForest Classifier ###############
print("------------------------------------------Task1----------------------------------------------")
print("Random Forest Classifier:")
print("\n")
classifier_random_forest(80,seed,x_train, y_train, x_test, y_test)
classifier_random_forest(110,seed,x_train, y_train, x_test, y_test)
classifier_random_forest(175,seed,x_train, y_train, x_test, y_test)

################ AdaBoost Classifier ################
print("AdaBoost Classifier:")
print("\n")

#Decision stump 1 with max_depth:1 and min_samples:1
decision_stump1 = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
decision_stump1 = decision_stump1.fit(x_train, y_train)
decision_stump1_accuracy = decision_stump1.score(x_test, y_test)

#Decision stump 1 with max_depth:1 and min_samples:1
decision_stump2 = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1)
decision_stump2 = decision_stump2.fit(x_train, y_train)
decision_stump2_accuracy = decision_stump2.score(x_test, y_test)

#AdaBoost classifiers with different hyperparameters
classifier_adaboost(100, 1, seed, x_train, y_train, x_test, y_test, decision_stump1,1)
classifier_adaboost(150, 1, seed, x_train, y_train, x_test, y_test, decision_stump1,1)
classifier_adaboost(175, 1, seed, x_train, y_train, x_test, y_test, decision_stump1,1)
classifier_adaboost(100, 1, seed, x_train, y_train, x_test, y_test, decision_stump2,2)
classifier_adaboost(150, 1, seed, x_train, y_train, x_test, y_test, decision_stump2,2)
classifier_adaboost(175, 1, seed, x_train, y_train, x_test, y_test, decision_stump2,2)


################################# Task 2 #############################################
print("------------------------------------------Task2----------------------------------------------")
print("Ensemble Classifier:")
print("\n")

#Neural Network
NN_classifier = MLPClassifier(random_state= seed)
NN_classifier = NN_classifier.fit(x_train, y_train)

# K Nearest Neighbours
KNN_classifier = KNeighborsClassifier(n_neighbors=10)
KNN_classifier = KNN_classifier.fit(x_train, y_train)

#Logistic Regression
LR_classifier = LogisticRegression(random_state= seed)
LR_classifier = LR_classifier.fit(x_train, y_train)

#Naives Bayes
NB_classifier = GaussianNB()
NB_classifier = NB_classifier.fit(x_train, y_train)

#Decision Tree
DT_classifier = DecisionTreeClassifier(max_depth=1, min_samples_leaf=1)
DT_classifier = DT_classifier.fit(x_train, y_train)

#printing the Accuracy and confusion Matrix for each matrix
for classifier, name in zip([NN_classifier, KNN_classifier, LR_classifier, NB_classifier, DT_classifier], ['Neural Networks', 'K Nearest Neighbours','Logistic Regression','Naive Bayes', 'Decision Tree']):
    y_pred = classifier.predict(x_test)
    print("Classifier: %s --> Accuracy: %0.4f " % (name, classifier.score(x_test, y_test)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\n")

# Ensemble classifier with Unweighted majority vote
ensemble_classifier = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier)], voting='hard')
ensemble_classifier = ensemble_classifier.fit(x_train, y_train)
y_pred = ensemble_classifier.predict(x_test)
print("Accuracy (unweighted majority voting): %0.4f" % (ensemble_classifier.score(x_test, y_test)))
print("\n")

# Ensemble classifier with Weighted Majority Voting 
##1. equal weights
weights = dict([("Neural Network", 1),("K-Nearest Neighbours", 1),("Logistic Regression", 1),("Naives Bayes", 1),("Decision Tree", 1) ])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted1 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier)], voting='soft',weights=[1,1,1,1,1])
ensemble_classifier_weighted1 = ensemble_classifier_weighted1.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted1.predict(x_test)
print("Accuracy (Equal weights): %0.4f" % (ensemble_classifier_weighted1.score(x_test, y_test)))
print("\n")
#
##2. weights proportional to classification accuracy
weights = dict([("Neural Network", 1),("K-Nearest Neighbours", 2),("Logistic Regression", 3),("Naives Bayes", 2),("Decision Tree", 2) ])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted2 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier)], voting='soft',weights=[1,2,4,3,2])
ensemble_classifier_weighted2 = ensemble_classifier_weighted2.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted2.predict(x_test)
print("Accuracy (weights proportional to the classification accuracy) : %0.4f" % (ensemble_classifier_weighted2.score(x_test, y_test)))
print("\n")
#
##2. different weights proportional to classification accuracy
weights = dict([("Neural Network", 3),("K-Nearest Neighbours", 5),("Logistic Regression", 9),("Naives Bayes", 7),("Decision Tree", 5) ])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted3 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier)], voting='soft',weights=[3,5,9,7,5])
ensemble_classifier_weighted3 = ensemble_classifier_weighted3.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted3.predict(x_test)
print("Accuracy (weights proportional to the classification accuracy): %0.4f" % (ensemble_classifier_weighted3.score(x_test, y_test)))
print("\n")

################################## Task 3 ##########################################
#Ensemble classifier with Unweighted majority vote including Random Forest and AdaBoost
print("------------------------------------------Task3----------------------------------------------")
print("Ensemble Classifier including RandomForest and AdaBoost:")
print('\n')
randomForest = classifier_random_forest(130, seed, x_train, y_train, x_test,y_test)
adaBoost = classifier_adaboost(150, 1, seed, x_train, y_train, x_test, y_test, decision_stump1,1)
for classifier, label in zip([randomForest, adaBoost], ['Random Forest', 'AdaBoost']):
    y_pred = classifier.predict(x_test)
    print("Classifier: %s --> Accuracy: %0.4f" % (label, classifier.score(x_test, y_test)))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

#Ensemble classifier with Unweighted majority vote
ensemble_classifier2 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier), ('model', randomForest), ('ada', adaBoost)], voting='hard')
ensemble_classifier2 = ensemble_classifier2.fit(x_train, y_train)
y_pred = ensemble_classifier2.predict(x_test)
print("Accuracy (unweighted majority voting): %0.4f" % (ensemble_classifier2.score(x_test, y_test)))
print("\n")

# Ensemble classifier with Weighted Majority Voting 
##1. equal weights
weights = dict([("Neural Network", 1),("K-Nearest Neighbours", 1),("Logistic Regression", 1),("Naives Bayes", 1),("Decision Tree", 1),("Random Forest",1), ("AdaBoost",1)])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted4 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[1,1,1,1,1,1,1])
ensemble_classifier_weighted4 = ensemble_classifier_weighted4.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted4.predict(x_test)
print("Accuracy (Equal weights): %0.4f" % (ensemble_classifier_weighted4.score(x_test, y_test)))
print("\n")
#
##2. different weights proportional to accuracy
weights = dict([("Neural Network", 4),("K-Nearest Neighbours", 5),("Logistic Regression", 7),("Naives Bayes", 6),("Decision Tree", 3),("Random Forest",3), ("AdaBoost",8)])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted5 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[4,5,7,6,3,3,8])
ensemble_classifier_weighted5 = ensemble_classifier_weighted5.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted5.predict(x_test)
print("Accuracy (weights proportional to the classification accuracy): %0.4f" % (ensemble_classifier_weighted5.score(x_test, y_test)))
print("\n")
#
##3. different weights proportional to accuracy
weights = dict([("Neural Network", 5),("K-Nearest Neighbours", 6),("Logistic Regression", 7),("Naives Bayes", 6),("Decision Tree", 5),("Random Forest",3), ("AdaBoost",9)])
print("weights\n")
print(weights)
print("\n")
ensemble_classifier_weighted6 = VotingClassifier(estimators=[('NN', NN_classifier),('KNN', KNN_classifier),('LR', LR_classifier),('NB', NB_classifier), ('DT', DT_classifier), ('rf', randomForest), ('ada', adaBoost)], voting='soft',weights=[5,6,7,6,5,3,9])
ensemble_classifier_weighted6 = ensemble_classifier_weighted6.fit(x_train, y_train)
y_pred = ensemble_classifier_weighted6.predict(x_test)
print("Accuracy (weights proportional to the classification accuracy): %0.4f" % (ensemble_classifier_weighted6.score(x_test, y_test)))
print("\n")
