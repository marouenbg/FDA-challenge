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
#add sex and disease as features
training.loc[:,'gender']=labels.iloc[:,1].values
training.loc[:,'msi']=labels.iloc[:,2].values

#More exploration for data imputation
#distribution of proteins by sex
#distribution of proteins by disease status
#impute median/mean/sample between mean and sd
#impute by row mean or col mean
#scale + standardize + fill missing
#Optimize hyperparameters

#feature selection though stability selection
nFeatures=20
rlasso = RandomizedLasso(alpha=0.01, sample_fraction=0.75, n_resampling=3000)#need high resampling to
rlasso.fit(training, matching.iloc[:,1])
names=list(training)
#print "Features sorted by their score:"
#print sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
vecSort=sorted(zip(map(lambda x: round(x, 4), rlasso.scores_), names), reverse=True)
colNameVecSort=[x[1] for x in vecSort]
selFeatures=colNameVecSort[0:nFeatures]#select 30 features
print vecSort[0]

#learn and cross validate : target is gender
clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0, class_weight="balanced") 
clf.fit(training.loc[:,selFeatures], matching)
scores = cross_val_score(clf, training.loc[:,selFeatures], matching.iloc[:,1], cv=5)
print scores.mean()

#y=clf.predict(training.iloc[misMatcInd,:].loc[:,selFeatures])   
#y==labels.iloc[misMatcInd,classToPredict]

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