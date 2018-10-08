import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier #call extra trees
import random #to set the seed
import os
from sklearn.model_selection import cross_val_score #cross validation
from sklearn.feature_selection import SelectFromModel #select top features
from xgboost import XGBClassifier #for xgboost

#change working directory
os.chdir('/home/marouen/challenges/FDA/data')

#read data files
training = pd.read_csv("train_pro.tsv", sep="\t")
training = training.T#transpose the dataframe
labels = pd.read_csv("train_cli.tsv", sep="\t")
matching = pd.read_csv("sum_tab_1.csv")

#Mesure size of classes
print(labels.groupby(['gender']).size())
print(labels.groupby(['msi']).size())

#Encode sex and MSI
labels['gender'].replace(['Female','Male'],[1,0], inplace=True)
labels['msi'].replace(['MSI-Low/MSS','MSI-High'],[0,1], inplace=True)

#data exploration
training.max().plot(kind='hist')
training.isna().sum().plot(kind='hist')
sum(matching["mismatch"])

#for now impute by mean per row (patient) 
#as mean by column (protein) yields entire NAN columns -> worth thr try anyway as feature selection
training=training.T.fillna(training.mean(axis=1)).T

#remove mismatches for now
indices = np.where(matching["mismatch"]==0)[0]
misMatcInd = np.where(matching["mismatch"]==1)[0]
trainingNoMismatch = training.iloc[indices,:]
labelsNoMismatch = labels.iloc[indices,:]

#More exploration for data imputation
#distribution of proteins by sex
#distribution of proteins by disease status
#impute median/mean/sample between mean and sd
#impute by row mean or col mean
#scale + standardize + fill missing
#Optimize hyperparameters

#feature selection though stability selection
trainingNoMismatch.loc[:,'add']=labelsNoMismatch.iloc[:,2].values
#random.seed(1)
classToPredict=1 #could be 1 (sex) or 2 (msi)
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0, class_weight="balanced")  #need high resampling to
clf.fit(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict]) #predict sex 1 disease 2
names=list(trainingNoMismatch)
#print "Features sorted by their score:"
#print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
vecSort=sorted(zip(map(lambda x: round(x, 4), clf.feature_importances_), names), reverse=True)
colNameVecSort=[x[1] for x in vecSort]
#print(colNameVecSort)

scoreVec=[]
featVec=[1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,30,40,50,60,70,80,90,100]
for nFeatures in featVec:
	print(nFeatures)
	selFeatures=colNameVecSort[0:nFeatures] #select nFeatures
	#print(selFeatures)
	scores = cross_val_score(clf, trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict], cv=5, scoring='balanced_accuracy')
	scoreVec.append(scores.mean())

print(list(zip(featVec,scoreVec)))


xgboot = XGBClassifier()
#y_pred = model.predict(X_test)
#y=clf.predict(training.iloc[misMatcInd,:].loc[:,selFeatures])   
#y==labels.iloc[misMatcInd,classToPredict]
