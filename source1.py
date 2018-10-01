#@ Author - Shobhit
#Date - 07-03-2018

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,ShuffleSplit,cross_val_score
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,accuracy_score,classification_report
from sklearn.externals import joblib
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import graphviz

#import matplotlib.pyplot as plt
#%matplotlib inline

#importing Dataset
def importData():
	data = pd.read_csv('FinalHearT.csv', sep=" ", header=None)
	data.columns = ["age", "sex", "cpt", "rbp", "chol", "fbs", "ecg", "hr", "eia", "oldpeak", "slope", "nov", "thal", "rafg1C", "rafg1M", "rafg1F", "rafg2C", "rafg2M", "rafg2F","outcome"]
	df = pd.DataFrame(data)
	vals_to_replace = {1:0, 2:1}
	df['outcome'] = df['outcome'].map(vals_to_replace)
	#print ("Shape:", data.shape)
	#print ("\nFeatures:", data.columns)
	#print feature_names,class_names
	#print df
	return df

#Splitting dataset
def splitDataset(data):
	X = data[data.columns[:-1]]
	y= data[data.columns[-1]]
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=179)
	#print(X_train.shape)
	#print(X_test.shape)
	#print(y_train.shape)
	#print(y_test.shape)
	#print("\n Feature matrix:\n",x.head())
	#print("\nResponse vector:\n", y.head())
	return X,y,X_train,X_test,y_train,y_test

def subsample(data, ratio=1.0):
	sample = list()
	n_sample = round(len(data) * ratio)
	while len(sample) < n_sample:
		index = randrange(len(data))
		sample.append(data[index])
	return sample
	
def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split	


def train_using_gini(X_train, X_test, y_train):
 
	# Creating the classifier object
	clf_gini = DecisionTreeClassifier(criterion = "gini",
	random_state = 100, min_samples_leaf=5)
 
	# Performing training
	clf_gini.fit(X_train, y_train)
	return clf_gini


def prediction(X_test, clf_object):
 
	# Predicton on test with giniIndex
	y_pred = clf_object.predict(X_test)
	print("Predicted values:")
	#print(y_pred)
	return y_pred

def cal_accuracy(y_test, y_pred):
	 
	print("Confusion Matrix: ",
	confusion_matrix(y_test, y_pred))
	 
	print ("Accuracy : ",
	accuracy_score(y_test,y_pred)*100)
	 
	print("Report : ",
	classification_report(y_test, y_pred)) 

def cal_rof_accuracy(y_test,clf,X_test):
	roc = roc_auc_score(y_test, clf.predict(X_test))
	print ("clf AUC = %2.2f" % roc)
	return roc


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
	cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
	clf_gini = train_using_gini(X_train,X_test,y_train)
	#visualize(clf_gini,feature_name,class_name,data)
	for n_trees in [1,10,50,100]:
		clf = BaggingClassifier(clf_gini, n_estimators = n_trees, max_samples=0.80)
		scores = cross_val_score(clf, X, y, verbose = 1, cv=cv, scoring='accuracy')
		clf.fit(X_train,y_train)
		print('Trees: %d' % n_trees)
		print('Scores: %s' % scores)
		print("Mean Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
				

		#cm_5=confusion_matrix(y_test,clf.predict(X_test),[1,2])
		#print cm_5
	#roc = cal_rof_accuracy(y_test,clf,X_test)
	#pr,tpr,thresholds = metrics.roc_curve(y_test,clf.predict(X_test))
	
	#joblib.dump(clf, 'source.pkl')
 

if __name__ =="__main__":
	main()	