import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier #call extra trees
from sklearn.linear_model import RandomizedLasso #feature selection
import random #to set the seed
import os
from sklearn.model_selection import cross_val_score #cross validation

#change working directory
os.chdir('/home/marouen/challenges/FDA/data')

#read data files
training = pd.read_csv("train_pro.tsv", sep="\t")
training = training.T#transpose the dataframe
labels = pd.read_csv("train_cli.tsv", sep="\t")
matching = pd.read_csv("sum_tab_1.csv")

#Mesure size of classes
print labels.groupby(['gender']).size()
print labels.groupby(['msi']).size()

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
#trainingNoMismatch.loc[:,'add']=labelsNoMismatch.iloc[:,1].values
#random.seed(1)
classToPredict=1#could be 1 or 2
if classToPredict==1:#sex
    nFeatures=30
elif classToPredict==2:#disease
    nFeatures=20
rlasso = RandomizedLasso(alpha=0.025,n_resampling=3000)#need high resampling to
rlasso.fit(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict])#predict sex 1 disease 2
names=list(trainingNoMismatch)
#print "Features sorted by their score:"
#print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
vecSort=sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
colNameVecSort=[x[1] for x in vecSort]
selFeatures=colNameVecSort[1:nFeatures]#select 30 features

#learn and cross validate : target is gender
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0, class_weight="balanced") 
scores = cross_val_score(clf, trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict], cv=5)
print scores.mean()
clf.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict])
y=clf.predict(training.iloc[misMatcInd,:].loc[:,selFeatures])   
y==labels.iloc[misMatcInd,classToPredict]

#disease
'''
1      True
9     False
12     True
38    False
46     True
57    False
64     True
67    False
71    False
77    False
78     True
79     True
'''

#sex
'''
1     False
9      True
12     True
38    False
46     True
57     True
64    False
67    False
71     True
77     True
78    False
79    False
'''