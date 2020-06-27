import numpy as np
from sklearn.model_selection import train_test_split, KFold
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import accuracy_score
from math import sqrt
from statistics import mode



print("Exercise #1")
dataTrain = np.loadtxt('IDSWeedCropTrain.csv',delimiter=',')
dataTest = np.loadtxt('IDSWeedCropTest.csv',delimiter=',')
xTrain = dataTrain[:,:-1]
yTrain = dataTrain[:,-1]
xTest = dataTest[:,:-1]
yTest = dataTest[:,-1]

############## 1 - Nearest Neighbour ##############

class knn1():
    def __init__(self, k = 1):
        self.k = k
    def fit(self, xtrain, ytrain):
        self.n_feat = xtrain.shape[1]
        self.n_dps = xtrain.shape[0]
        self.xtrain = xtrain
        self.ytrain = ytrain
        print("K = 1 KNN Model Fitted.")
    def predict(self, xtest):
        xtest = np.array(xtest)
        assert xtest.shape[1] == self.n_feat
        self.predictions = []
        for dp in xtest:
            dp_distances = []
            for i in range(self.n_dps):
                dist = 0
                for j in range(self.n_feat):
                    dist = dist + (dp[j] - self.xtrain[i][j])**2
                dist = sqrt(dist)
                dp_distances.append(dist)
            dp_nn = np.argmin(dp_distances)
            dp_predict = self.ytrain[dp_nn]
            self.predictions.append(dp_predict)
        print("Predictions made.")
        return self.predictions
    def acc_test(self, yTest):
        self.accuracy = accuracy_score(yTest, self.predictions)
        print("Accuracy: ", self.accuracy, "\nClassification Error: ", 1-self.accuracy)
        
        
    


weed_knn = knn1()
weed_knn.fit(xTrain, yTrain)
print("Prediction performance on training data:")
weed_knn.predict(xTrain)
weed_knn.acc_test(yTrain)
print("Prediction performance on test data:")
weed_knn.predict(xTest)
weed_knn.acc_test(yTest)


############## K - Nearest Neighbours ##############

class knn():
    def __init__(self, k = 1):
        self.k = k
    def fit(self, xtrain, ytrain):
        self.n_feat = xtrain.shape[1]
        self.n_dps = xtrain.shape[0]
        self.xtrain = xtrain
        self.ytrain = ytrain
#         print("KNN Model Fitted.")
    def predict(self, xtest):
        xtest = np.array(xtest)
        assert xtest.shape[1] == self.n_feat
        self.predictions = []
        for dp in xtest:
            dp_distances = []
            for i in range(self.n_dps):
                dist = 0
                for j in range(self.n_feat):
                    dist = dist + (dp[j] - self.xtrain[i][j])**2
                dist = sqrt(dist)
                dp_distances.append(dist)
            dp_nns = np.argsort(dp_distances)[:self.k]
            dp_predict = mode(self.ytrain[dp_nns]) 
            self.predictions.append(dp_predict)
#         print("Predictions made.")
        return self.predictions
    def acc_test(self, yTest):
        self.accuracy = accuracy_score(yTest, self.predictions)
        print("Accuracy test score:", self.accuracy, "\nClassification error:", 1-self.accuracy)


############## K-fold cross-validation ##############

print("Exercise #2")
def k_fold(xTrain, yTrain, nsplits = 5):
    k_list = [k for k in range(1,13,2)]
    cv = KFold(n_splits = nsplits)
    accuracy_scores = {}
    accuracy_means = {}
    for j in k_list:
        
        accuracy_scores[j] = []
        for train, test in cv.split(xTrain):
            xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train], xTrain[test], yTrain[train], yTrain[test]
            weed_knn = knn(k=j)
            weed_knn.fit(xTrainCV,yTrainCV)
            weed_predictions = weed_knn.predict(xTestCV)

            accscore = accuracy_score(yTestCV, weed_predictions)
            accuracy_scores[j].append(accscore)
        print("Predictions made for k =", j)
        accuracy_means[j] = np.mean(accuracy_scores[j])
    k_best = list(accuracy_means.keys())[np.argmax(list(accuracy_means.values()))]
    print("Accuracy Scores: ", accuracy_scores, "\nAccuracy Means: ", accuracy_means, "\nk with the best accuracy: ", k_best)
    return k_best

k_best = k_fold(xTrain, yTrain)

print("Exercise #3")
weed_knn = knn(k=k_best)
weed_knn.fit(xTrain, yTrain)
print("Prediction performance on training data, trained on unnormalised training data and using k-best:")
weed_knn.predict(xTrain)
weed_knn.acc_test(yTrain)
print("Prediction performance on test data, trained on unnormalised training data and using k-best:")
weed_knn.predict(xTest)
weed_knn.acc_test(yTest)

############## Hyperparameter tuning, find the best k for the dataset ##############

print("Exercise #4")
class knn_best():
    def __init__(self, xtrain, ytrain, max_k=11):
        self.max_k = max_k
        self.n_feat = xtrain.shape[1]
        self.n_dps = xtrain.shape[0]
        self.xtrain = xtrain
        self.ytrain = ytrain
    
    def normalise(self):
        self.scalar = preprocessing.StandardScaler().fit(self.xtrain)
        self.normalisedx = self.scalar.transform(self.xtrain)
        
    def k_fold(self, nsplits = 5):
        k_list = [k for k in range(1,self.max_k+1,2)]
        cv = KFold(n_splits = nsplits)
        accuracy_scores = {}
        accuracy_means = {}
        classification_error_means = {}        
        yTrain = self.ytrain 
        try: #optional use of normalised data, by checking whether normalise function has been run.
            xTrain = self.normalisedx
        except AttributeError:
            xTrain = self.xtrain
            
        for j in k_list:
            accuracy_scores[j] = []
            classification_error_means[j] = []
            for train, test in cv.split(xTrain):
                xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train], xTrain[test], yTrain[train], yTrain[test]
                
                weed_knn = knn(k=j)
                weed_knn.fit(xTrainCV,yTrainCV)
                weed_predictions = weed_knn.predict(xTestCV)
                accscore = accuracy_score(yTestCV, weed_predictions)
                accuracy_scores[j].append(accscore)
            accuracy_means[j] = np.mean(accuracy_scores[j])
            classification_error_means[j] = 1-accuracy_means[j]
            print("Predictions made for k =", j)
        self.k_best = list(accuracy_means.keys())[np.argmax(list(accuracy_means.values()))]
        print("\nMean accuracy: ", accuracy_means, "\nMean classification error: ", classification_error_means, "\nBest k: ", self.k_best)
        return self.k_best     
            
    def predict(self, xTest, yTest):
        weed_knn = knn(k=self.k_best)
        try: #optional use of normalised data, by checking whether normalise function has been run.
            xTrain = self.normalisedx
            xTest = self.scalar.transform(xTest)
        except AttributeError:
            xTrain = self.xtrain
        weed_knn.fit(xTrain, self.ytrain)
        weed_predictions = weed_knn.predict(xTest)
        self.accuracy = accuracy_score(yTest, weed_predictions)
        print("Accuracy score: ", self.accuracy,"\nClassification error: ", 1-self.accuracy)
        return self.accuracy
    
knn_best = knn_best(xTrain, yTrain)
knn_best.normalise()
knn_best.k_fold()
print("Prediction performance on training data, trained on normalised training data and using k-best:")
knn_best.predict(xTrain,yTrain)
print("Prediction performance on test data, trained on normalised training data and using k-best:")
knn_best.predict(xTest, yTest)

