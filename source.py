#@ Author - Shobhit
#Date - 07-03-2018

import pandas as pd
from sklearn import datasets, linear_model
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import graphviz
import matplotlib.pyplot as plt
#%matplotlib inline

#importing Dataset
def importData():
	data = pd.read_csv('FinalHearT.csv', sep=" ", header=None)
	data.columns = ["age", "sex", "cpt", "rbp", "chol", "fbs", "ecg", "hr", "eia", "oldpeak", "slope", "nov", "thal", "rafg1C", "rafg1M", "rafg1F", "rafg2C", "rafg2M", "rafg2F","outcome"]
	df = pd.DataFrame(data)
	#print ("Shape:", data.shape)
	#print ("\nFeatures:", data.columns)
	#print feature_names,class_names
	#print df
	return data

#Splitting dataset
def splitDataset(data):
	X = data[data.columns[:-1]]
	y= data[data.columns[-1]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
	#print(X_train.shape)
	#print(X_test.shape)
	#print(y_train.shape)
	#print(y_test.shape)
	#print("\n Feature matrix:\n",x.head())
	#print("\nResponse vector:\n", y.head())
	return X,y,X_train,X_test,y_train,y_test

def train_using_gini(X_train, X_test, y_train):
 
	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",
	random_state = 100,max_depth=3, min_samples_leaf=5)
 
	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini

def prediction(X_test, clf_object):
 
	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	print(y_pred)
	return y_pred

def cal_accuracy(y_test, y_pred):
	 
	print("Confusion Matrix: ",
	confusion_matrix(y_test, y_pred))
	 
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	 
	print("Report : ",
	classification_report(y_test, y_pred)) 

def visualize(clf_gini,feature_name,class_name,data):
	dot_data = tree.export_graphviz(clf_gini, out_file=None,
		 feature_names=feature_name,  
                         class_names=class_name,  
                         filled=True, rounded=True,  
                         special_characters=True) 
	graph = graphviz.Source(dot_data) 
	graph.render("data")
	graph 	  

def main():
	data = importData()
	feature_name = list(data.columns[:-1])
	class_name = ['1','2']
	X, y, X_train, X_test,y_train,y_test = splitDataset(data)
	clf_gini = train_using_gini(X_train, X_test, y_train)
	visualize(clf_gini,feature_name,class_name,data)
	#print(X_test)
	y_pred_gini = prediction(X_test, clf_gini)
	cal_accuracy(y_test, y_pred_gini)

 

if __name__ =="__main__":
	main()	