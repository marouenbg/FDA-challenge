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
from sklearn.linear_model import RandomizedLasso #stability selection

#change working directory
os.chdir('/home/marouen/challenges/FDA/data')

#Random seed
seed=42
random.seed(seed)

#read data files
training  = pd.read_csv("train_pro.tsv", sep="\t")
training  = training.T #transpose the dataframe
labels    = pd.read_csv("train_cli.tsv", sep="\t")
matching  = pd.read_csv("sum_tab_1.csv")
test      = pd.read_csv("test_pro.tsv", sep="\t")
test      = test.T
labelsTest= pd.read_csv("test_cli.tsv", sep="\t") 

#Mesure size of classes
#print(labels.groupby(['gender']).size())
#print(labels.groupby(['msi']).size())

#Encode sex and MSI
labels['gender'].replace(['Female','Male'],[0,1], inplace=True)
labels['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)
labelsTest['gender'].replace(['Female','Male'],[0,1], inplace=True)
labelsTest['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)
#print(sum(labels[labels['gender'] == 1]['msi'])/sum(labels['gender'] == 1))
#print(sum(labels[labels['gender'] == 0]['msi'])/sum(labels['gender'] == 0))
#print(sum(labels[labels['msi'] == 1]['gender'])/sum(labels['msi'] == 1))
#print(sum(labels[labels['msi'] == 0]['gender'])/sum(labels['msi'] == 0))

#data exploration
training.max().plot(kind='hist')
training.isna().sum().plot(kind='hist')
sum(matching["mismatch"])

#remove mismatches for now
indices = np.where(matching["mismatch"]==0)[0]
misMatchInd = np.where(matching["mismatch"]==1)[0]
trainingNoMismatch = training.iloc[indices,:]
labelsNoMismatch = labels.iloc[indices,:]

impute='mean' # mean or 'specific' 
if impute=='mean':
	#for now impute by mean per row (patient) 
	#as mean by column (protein) yields entire NAN columns -> worth thr try anyway as feature selection
	#training=training.T.fillna(training.mean(axis=1)).T
	#combined
	result=pd.concat([training,test])
	result=result.fillna(result.mean(axis=1))
	result.dropna(axis=1, inplace=True)
	result=(result-result.mean())/result.std()
	training=result.iloc[:80,:]
	test    =result.iloc[80:,:]
	#by protein
	#training=training.fillna(training.mean(axis=1))
	#training.dropna(axis=1, inplace=True)
	#normalize
	#training=(training-training.mean())/training.std()
	#Test
	#test=test.fillna(test.mean(axis=1))
	#test.dropna(axis=1, inplace=True)
	#test=(test-test.mean())/test.std()
elif impute=='specific':
	#identify x and y proteins
	r=re.compile(".+(X|Y)")
	#prot      =list(filter(r.match,list(training))) #matching XY proteins
	prot      =['USP9Y']
	msiProt   =list(set(list(training)) - set(prot)) #non matching prot
	maleInd   =list( set(np.where(labels["gender"]==1)[0]) & set(indices))
	femaleInd =list( set(np.where(labels["gender"]==0)[0]) & set(indices))
	msiLoInd  =list( set(np.where(labels["msi"]==1)[0]) & set(indices))
	msiHiInd  =list( set(np.where(labels["msi"]==0)[0]) & set(indices))
	meanMale  =np.mean(training.iloc[maleInd,:].loc[:,prot])
	meanFemale=np.mean(training.iloc[femaleInd,:].loc[:,prot])
	meanMsiLo =np.mean(training.iloc[msiLoInd,:].loc[:,msiProt])
	meanMsiHi =np.mean(training.iloc[msiHiInd,:].loc[:,msiProt])
	for i in range(len(prot)):
		training.loc[:,prot[i]].iloc[maleInd]=training.loc[:,prot[i]].iloc[maleInd].fillna(meanMale[i])
		training.loc[:,prot[i]].iloc[femaleInd]=training.loc[:,prot[i]].iloc[femaleInd].fillna(meanFemale[i])
	#Impute the other missing by the mean (consider MSI and further refien prot liek CYP and COX)
	#for i in range(len(msiProt)):
		#training.loc[:,msiProt[i]].iloc[msiLoInd]=training.loc[:,msiProt[i]].iloc[msiLoInd].fillna(meanMsiLo[i])
		#training.loc[:,msiProt[i]].iloc[msiHiInd]=training.loc[:,msiProt[i]].iloc[msiHiInd].fillna(meanMsiHi[i])
	#Although we can't fill all proteins because some of the proteins in match are completely
	#na so fi++5614ll remaining (incl Mismatch) by mean of patient
	a=np.where(meanMale.isnull()==True)
	b=np.where(meanFemale.isnull()==True)
	c=np.where(meanMsiLo.isnull()==True)
	d=np.where(meanMsiHi.isnull()==True)
	e=reduce(np.union1d, (a,b,c,d))
	#delete rows of missing value
	training.drop(training.columns[e], axis=1, inplace=True)
	training=training.fillna(training.mean(axis=1))
	training.dropna(axis=1, inplace=True)
	#BONUS: do all of the above in pandas

#remove mismatches for now
indices = np.where(matching["mismatch"]==0)[0]
misMatchInd = np.where(matching["mismatch"]==1)[0]
trainingNoMismatch = training.iloc[indices,:]
labelsNoMismatch = labels.iloc[indices,:]

#More exploration for data imputation
#distribution of proteins by sex
#distribution of proteins by disease status
#impute median/mean/sample between mean and sd
#impute by row mean or col mean
#scale + standardize + fill missing
#Optimize hyperparameters
#new idea to find x chr prot less than 50 missing

compMat=np.zeros([len(labelsTest),4])
res=labelsTest.iloc[:,:1]
res["mismatch"]=0
compMat[:,2:4]=labelsTest.iloc[:,1:3]
for anteClass in [2,1]: #2 is predicting sex,1 is predicting msi
	if anteClass==1:
		classToPredict=2
		nFeat=7
	elif anteClass==2:
		classToPredict=1
		nFeat=15
	#feature selection though stability selection
	#classToPredict could be 1 (sex) or 2 (msi)
	X_train, X_test, y_train, y_test = train_test_split(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict], test_size=0.2, random_state=seed) #42
	rl = RandomizedLasso(alpha='aic', random_state=seed, n_resampling=1000)  #need high resampling to
	rl.fit(X_train,y_train) #predict sex 1 disease 2
	names=list(trainingNoMismatch)
	#print "Features sorted by their score:"
	#print(sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True))
	vecSort=sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True)
	colNameVecSort=[x[1] for x in vecSort]
	#print('add' in colNameVecSort)

	scoreVec=[]
	featVec=[1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,30,40,50,60,70,80,90,100]
	X_train, X_test, y_train, y_test = train_test_split(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict], test_size=0.2, random_state=42) #42
	for nFeatures in featVec:
		selFeatures=colNameVecSort[0:nFeatures] #select nFeatures
		sumneg=sum(y_train==0)
		sumpos=sum(y_train==1)
		xgboost = XGBClassifier(scale_pos_weight=sumneg/sumpos)	
		xgboost.fit(X_train.loc[:,selFeatures], y_train)
		y_pred=xgboost.predict(X_test.loc[:,selFeatures])
		a=accuracy_score(y_test,y_pred)
		scoreVec.append(a)

	print(list(zip(featVec,scoreVec)))

	#Predict
	rl.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict]) #predict sex 1 disease 2
	vecSort=sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True)
	colNameVecSort=[x[1] for x in vecSort]
	selFeatures=colNameVecSort[0:nFeatures]
	xgboost = XGBClassifier()
	xgboost.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict])
	y_pred=xgboost.predict(test.loc[:,selFeatures])
	compMat[:,classToPredict-1]=y_pred

def printResult(y_pred,compMat,res):
	for i in range(len(y_pred)):
		if(compMat[i,0] != compMat[i,2]) & (compMat[i,1] != compMat[i,3]):
			res.iloc[i,1]=1

	#write file:
	res.to_csv('submission1.csv',sep=',',index=False)

printResult(y_pred,compMat,res)
#To do:
#How to combine binary and continous features
#compute metric between pred and exp
#combine multipe metrics
#kappa cohen
#fill gender specific proteins by gender specific means

#Problems:
#not same set of features selected
