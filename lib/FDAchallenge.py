import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier #call extra trees
import random #to set the seed
import os
import re #regex
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.feature_selection import SelectFromModel #select top features
from xgboost import XGBClassifier #for xgboost
from sklearn.ensemble import RandomForestClassifier #random forest
from sklearn.model_selection import train_test_split #test set
from sklearn.model_selection import cross_val_predict #cross val prediction
from sklearn.metrics import accuracy_score #accuracy
from functools import reduce #for set union

#change working directory
os.chdir('/home/marouen/challenges/FDA/data')

#read data files
training = pd.read_csv("train_pro.tsv", sep="\t")
training = training.T#transpose the dataframe
labels = pd.read_csv("train_cli.tsv", sep="\t")
matching = pd.read_csv("sum_tab_1.csv")

#Mesure size of classes
#print(labels.groupby(['gender']).size())
#print(labels.groupby(['msi']).size())

#Encode sex and MSI
labels['gender'].replace(['Female','Male'],[0,1], inplace=True)
labels['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)
#print(sum(labels[labels['gender'] == 1]['msi'])/sum(labels['gender'] == 1))
#print(sum(labels[labels['gender'] == 0]['msi'])/sum(labels['gender'] == 0))
#print(sum(labels[labels['msi'] == 1]['gender'])/sum(labels['msi'] == 1))
#print(sum(labels[labels['msi'] == 0]['gender'])/sum(labels['msi'] == 0))

#data exploration
training.max().plot(kind='hist')
training.isna().sum().plot(kind='hist')
sum(matching["mismatch"])

impute='mean' # mean or 'specific' 
if impute=='mean':
	#for now impute by mean per row (patient) 
	#as mean by column (protein) yields entire NAN columns -> worth thr try anyway as feature selection
	#training=training.T.fillna(training.mean(axis=1)).T
	#by protein
	training=training.fillna(training.mean(axis=1))
	training.dropna(axis=1, inplace=True)
	#normalize
	#training=(training-training.mean())/training.std()
elif impute=='specific':
	#identify x and y proteins
	indices=np.where(matching["mismatch"]==0)[0]
	r=re.compile(".+(X|Y)")
	prot      =list(filter(r.match,list(training))) #matching XY proteins
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
	#na so fill remaining (incl Mismatch) by mean of patient
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

compMat=np.zeros([len(misMatchInd),4])
compMat[:,2:4]=labels.iloc[misMatchInd,1:3]
for anteClass in [2,1]: #2 is predicting sex,1 is predicting msi
	if anteClass==1:
		classToPredict=2
		nFeat=8
	elif anteClass==2:
		classToPredict=1
		nFeat=10
	#feature selection though stability selection
	#trainingNoMismatch['add1']=labelsNoMismatch.iloc[:,anteClass].values
	#trainingNoMismatch['add2']=1 - labelsNoMismatch.iloc[:,anteClass].values
	#trainingNoMismatch=(trainingNoMismatch-trainingNoMismatch.mean())/trainingNoMismatch.std()
	#print(trainingNoMismatch.loc[:,'add1'])
	#random.seed(1)
	#classToPredict could be 1 (sex) or 2 (msi)
	clf = ExtraTreesClassifier(n_estimators=100, class_weight="balanced")  #need high resampling to
	clf.fit(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict]) #predict sex 1 disease 2
	names=list(trainingNoMismatch)
	#print "Features sorted by their score:"
	#print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
	vecSort=sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True)
	colNameVecSort=[x[1] for x in vecSort]
	#print('add' in colNameVecSort)

	scoreVec=[]
	featVec=[1,2,3,4,5,6,7,8,9,10,15] #,16,17,18,19,20,30,40,50]
	#X_train, X_test, y_train, y_test = train_test_split(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict], test_size=0.4, random_state=42) #42
	for nFeatures in featVec:
		#print(nFeatures)
		selFeatures=colNameVecSort[0:nFeatures] #select nFeatures
		scores = cross_val_score(clf, trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict], cv=5, scoring='accuracy')
		#scores = cross_val_score(clf, X_train.loc[:,selFeatures], y_train, cv=3, scoring='f1_weighted')
		scoreVec.append(scores.mean())

	print(list(zip(featVec,scoreVec)))

	#Call xgboost
	sumneg=sum(labelsNoMismatch.iloc[:,classToPredict]==0)
	sumpos=sum(labelsNoMismatch.iloc[:,classToPredict]==1)
	selFeatures=colNameVecSort[0:nFeat]
	print(selFeatures)
	clf = ExtraTreesClassifier(n_estimators=200, class_weight="balanced")  #need high resampling to
	clf.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict])
	#print(y_test)
	#print(y_pred)
	#a=accuracy_score(y_test,y_pred)
	#print(a)

	#predict
	X_test = training.iloc[misMatchInd,:].loc[:,selFeatures]   
	y_pred = clf.predict(X_test)
	print(y_pred)
	#test
	compMat[:,classToPredict-1]=y_pred

print(compMat)

#TO dO:
#How to combine binary and continous features
#compute metric between pred and exp
#combine multipe metrics
#kappa cohen
#fill gender specific proteins by gender specific means
