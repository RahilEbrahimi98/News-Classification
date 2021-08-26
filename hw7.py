import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, precision_score, f1_score, recall_score
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn.model_selection import RepeatedKFold


def addLable():
    """add column 'FakeOrReal' to data of each file and set its value to 0 for fake news and 1 for real news."""
    real = pd.read_csv("pol-real.csv")
    real['FakeOrReal'] = 1
    real.to_csv("merged.csv", index= False)

    fake = pd.read_csv("pol-fake.csv")
    fake['FakeOrReal'] = 0
    fake.to_csv("fake.csv", index= False)

def merge():
    """add fake news data to real news which is already stored in 'merges.csv'. """
    merged = open("merged.csv", "a")
    fake = open("fake.csv", "r")
    i = 0
    for row in fake:
        if (i != 0): #ignore the first line which is the columns names
            merged.write(row)
        i = i + 1
    fake.close()
    merged.close()

def extractAndShuffledData():
    """extract feature data, target data and column names."""
    featureData = []
    targetData = []
    labels = (pd.read_csv("merged.csv", header=None).values)[0] #column names
    allData = (pd.read_csv("merged.csv")).values.tolist() #all data( feature data and target data)
    random.shuffle(allData) #shuffle the list 'allData'
    for i in range(len(allData)): #split feature data and target data
        featureData.append(allData[i][:-1])
        targetData.append(allData[i][-1])

    return featureData, targetData, labels

def plotTree(name, clf, labels): #name: name of image file for our tree
    fig = plt.figure(figsize=(50, 50))
    a = tree.plot_tree(clf, feature_names = labels, class_names=['fake', 'real'], filled=True)
    #plt.show()
    fig.savefig(name)


def crossValidation(X, y, depths):
    cv_scores_mean = []
    cv_scores_mean2 = []
    for depth in depths: # we are creating two models. one with gini method and the other with entropy.
        tree_model = DecisionTreeClassifier(max_depth = depth, criterion='gini')
        tree_model2 = DecisionTreeClassifier(max_depth=depth, criterion='entropy')
        cv_scores = cross_val_score(tree_model, X, y, cv = 10, scoring = 'accuracy')
        cv_scores2 = cross_val_score(tree_model2, X, y, cv=10, scoring='accuracy')
        cv_scores_mean.append(cv_scores.mean())
        cv_scores_mean2.append(cv_scores2.mean())
        print("mean accuracy with gini in depth ", depth, ": ", cv_scores_mean[-1])
        print("mean accuracy with entropy in depth ", depth, ": ", cv_scores_mean2[-1])
    maxAcc = max(cv_scores_mean)
    maxAcc2 = max(cv_scores_mean2)
    bestDepth = cv_scores_mean.index(maxAcc) + 5
    bestDepth2 = cv_scores_mean2.index(maxAcc2) + 5
    print("best accuracy with gini at depth: ", bestDepth, " with accuracy: ", maxAcc)
    print("best accuracy with entropy at depth: ", bestDepth2, " with accuracy: ", maxAcc2)


def q1():
    addLable()
    merge()
    x, y, labels = extractAndShuffledData() # x: feature data.  y: target data.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #x_train = 80% of feature data. y_train = target data for x_train
    #x_test = 20% of feature data. y_test = target data for x_test

    return x_train, x_test, y_train, y_test, labels, x, y

def q2(x_train, x_test, y_train, y_test, labels):
    print("----------------------q2----------------------")
    clf = DecisionTreeClassifier(criterion='gini')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("accuracy: \n", clf.score(x_test, y_test))
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("precision score: \n", precision_score(y_test, y_pred))
    print("recall score: \n", recall_score(y_test, y_pred))
    print("f1 score: \n", f1_score(y_test, y_pred))
    plotTree('q2_tree.png', clf, labels)

def q3(x_train, x_test, y_train, y_test, labels):
    print("----------------------q3----------------------")
    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print("accuracy: \n", clf.score(x_test, y_test))
    print("confusion matrix: \n", confusion_matrix(y_test, y_pred))
    print("precision score: \n", precision_score(y_test, y_pred))
    print("recall score: \n", recall_score(y_test, y_pred))
    print("f1 score: \n", f1_score(y_test, y_pred))
    plotTree('q3_tree.png', clf, labels)

def q4(x, y):
    print("----------------------q4----------------------")
    rng = range(5, 21)
    crossValidation(x, y, rng)

def q6(x_train, x_test, y_train, y_test):
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)
    acc = clf.score(x_test, y_test)
    print("----------------------q6----------------------")
    print("accuracy is: ", acc)






x_train, x_test, y_train, y_test, labels, x, y= q1()

q2(x_train, x_test, y_train, y_test, labels)
q3(x_train, x_test, y_train, y_test, labels)
q4(x, y)
q6(x_train, x_test, y_train, y_test)