import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier #call extra trees
import random #to set the seed
import os
import re #regex
import math
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.feature_selection import SelectFromModel #select top features
from xgboost import XGBClassifier #for xgboost
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.model_selection import train_test_split #test set
from sklearn.model_selection import cross_val_predict #cross val prediction
from sklearn.metrics import accuracy_score #accuracy
from functools import reduce #for set union
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import f1_score #f1 score
from sklearn.linear_model import RandomizedLasso #stability selection
from sklearn import preprocessing
import xgboost as xgb #xgboost python api
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from itertools import product
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV #Optimize hyperparameters
from sklearn.model_selection import StratifiedKFold
from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform
import pickle #for saving xgboost models
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials #fmin
import gc #fmin
from sklearn.model_selection import RepeatedStratifiedKFold #repeat kfold

#change working directory
os.chdir('../data')

#Random seed
seed=142 #142 for the 6 first param set pf xgboost
random.seed(seed)

#read data files
def loadData():
        training  = pd.read_csv("train_pro.tsv", sep="\t")
        training  = training.T #transpose the dataframe
        labels    = pd.read_csv("train_cli.tsv", sep="\t")
        matching  = pd.read_csv("sum_tab_1.csv")
        test      = pd.read_csv("test_pro.tsv", sep="\t")
        test      = test.T
        labelsTest= pd.read_csv("test_cli.tsv", sep="\t") 
        return training,labels,matching,test,labelsTest

def encodeData(labels,labelsTest):
        #Encode sex and MSI
        labels['gender'].replace(['Female','Male'],[0,1], inplace=True)
        labels['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)
        labelsTest['gender'].replace(['Female','Male'],[0,1], inplace=True)
        labelsTest['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)
        return labels,labelsTest

def imputeMissing(impute,training,test): 
	if impute=='mean':
		#combined
		result=pd.concat([training,test])
		#fillna of y proteins by zero
		result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]]=result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]].fillna(0) #USP9
		#fill the rest by mean
		result=result.fillna(result.mean()) # -1
		result.dropna(axis=1, inplace=True, how='all')
		#result=(result-result.mean())/result.std()
		training=result.iloc[:80,:]
		test    =result.iloc[80:,:]
	elif impute=='-1butgender':
		result=pd.concat([training,test])
		#fillna of y proteins by zero
		result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]]=result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]].fillna(0) #USP9
		#fill the rest by mean
		result=result.fillna(-1)
		result.dropna(axis=1, inplace=True, how='all')
		#result=(result-result.mean())/result.std()
		training=result.iloc[:80,:]
		test    =result.iloc[80:,:]
	elif impute=='-1':
                result=pd.concat([training,test])
                #fillna of y proteins by zero
                #fill the rest by mean
                result=result.fillna(-1)
                result.dropna(axis=1, inplace=True, how='all')
                #result=(result-result.mean())/result.std()
                training=result.iloc[:80,:]
                test    =result.iloc[80:,:]
	return training,test

def predictCompetition(trainingNoMismatch,labelsNoMismatch,classToPredict,compMat,labelsTest,params):
        #rl.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict]) #pre$
        #vecSort=sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True)
        #colNameVecSort=[x[1] for x in vecSort]
        #selFeatures=colNameVecSort[0:nFeatures]
	sumneg=sum(labelsNoMismatch.iloc[:,classToPredict]==0)
	sumpos=sum(labelsNoMismatch.iloc[:,classToPredict]==1)
	params['missing']=-1
	params['scale_pos_weight']=sumneg/sumpos
	print(params)
	xgboost = XGBClassifier(**params)
	xgboost.fit(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict])
	y_pred=xgboost.predict(test)
	probs=xgboost.predict_proba(test)
	#print(y_pred)
	#print(probs)
	compMat[:,classToPredict-1]=y_pred

def printResult(compMat,labelsTest,submission):
	res=labelsTest.iloc[:,:1]
	res["mismatch"]=0
	for i in range(compMat.shape[0]):
		if(compMat[i,1] != compMat[i,3]): # | (compMat[i,1] != compMat[i,3]):
			res.iloc[i,1]=1
	#print(compMat)
	print(sum(res['mismatch']))
	#write file:
	res.to_csv('./predictions/'+submission+'/submission'+submission+'.csv',sep=',',index=False)

for submission in [1,2]:
	#remove mismatches for now
	training,labels,matching,test,labelsTest=loadData()
	if submission in [1,2]:
		impute='mean'
	elif submission=='undefined':
		impute='-1butgender'
	else:
		impute='-1'

	training,test=imputeMissing(impute, training, test)
	#fill missing
	labels,labelsTest=encodeData(labels,labelsTest)
	indices = np.where(matching["mismatch"]==0)[0]
	misMatchInd = np.where(matching["mismatch"]==1)[0]
	trainingNoMismatch = training.iloc[indices,:]
	labelsNoMismatch = labels.iloc[indices,:]

	#generate prediction data
	compMat=np.zeros([len(labelsTest),4])
	compMat[:,2:4]=labelsTest.iloc[:,1:3]

	for classToPredict in [1,2]:
		if submission==1:
			folder1='11'
			folder2='11'
		elif submission==2:
			folder1='6'
			folder2='6'
		if classToPredict==1:
			params=pd.read_csv("../data/IntParams/"+folder1+"/xgb-random-grid-search-results1.csv")
		else:
			params=pd.read_csv("../data/IntParams/"+folder2+"/xgb-random-grid-search-results2.csv")

		params=params.to_dict('records')[0]
		params.pop('bestF1',None)
		params.pop('Unnamed: 0',None)
		predictCompetition(trainingNoMismatch,labelsNoMismatch,classToPredict,compMat,labelsTest,params)

	#print results
	printResult(compMat,labelsTest,str(submission))


