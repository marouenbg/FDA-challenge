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

#change working directory
os.chdir('/home/marouen/challenges/FDA/data')

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
	#print(sum(labels[labels['gender'] == 1]['msi'])/sum(labels['gender'] == 1))
	#print(sum(labels[labels['gender'] == 0]['msi'])/sum(labels['gender'] == 0))
	#print(sum(labels[labels['msi'] == 1]['gender'])/sum(labels['msi'] == 1))
	#print(sum(labels[labels['msi'] == 0]['gender'])/sum(labels['msi'] == 0))
	return labels,labelsTest

def imputeMissing(impute,training,test): 
	if impute=='mean':
		#for now impute by mean per row (patient) 
		#as mean by column (protein) yields entire NAN columns -> worth thr try anyway as feature selection
		#training=training.T.fillna(training.mean(axis=1)).T
		#combined
		result=pd.concat([training,test])
		#fillna of y proteins by zero
		result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]]=result.loc[:,["RPS4Y1","RPS4Y2","EIF1AY","DDX3Y"]].fillna(0) #USP9
		#fill the rest by mean
		result=result.fillna(result.mean()) #-1
		result.dropna(axis=1, inplace=True, how='all')
		#result=(result-result.mean())/result.std()
		training=result.iloc[:80,:]
		test    =result.iloc[80:,:]
	elif impute=='specific':
		#identify x and y proteins
		r=re.compile(".+Y")
		prot      =list(filter(r.match,list(training))) #matching XY proteins
		print(prot)
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
	return training,test

def predictCompetition(rl,trainingNoMismatch,labelsNoMismatch,classToPredict,names,compMat,labelsTest):
	compMat=np.zeros([len(labelsTest),4])
	compMat[:,2:4]=labelsTest.iloc[:,1:3]
	rl.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict]) #predict sex 1 disease 2
	vecSort=sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True)
	colNameVecSort=[x[1] for x in vecSort]
	selFeatures=colNameVecSort[0:nFeatures]
	sumneg=sum(labelsNoMismatch.iloc[:,classToPredict]==0)
	sumpos=sum(labelsNoMismathc.iloc[:,classToPredict]==1)
	xgboost = XGBClassifier(scale_pos_weight=sumneg/sumpos)
	xgboost.fit(trainingNoMismatch.loc[:,selFeatures], labelsNoMismatch.iloc[:,classToPredict])
	y_pred=xgboost.predict(test.loc[:,selFeatures])
	compMat[:,classToPredict-1]=y_pred

def printResult(y_pred,compMat,labelsTest):
	res=labelsTest.iloc[:,:1]
	res["mismatch"]=0
	for i in range(len(y_pred)):
		if(compMat[i,0] != compMat[i,2]) & (compMat[i,1] != compMat[i,3]):
			res.iloc[i,1]=1

	#write file:
	res.to_csv('submission1.csv',sep=',',index=False)

def classifier(X_train,y_train,X_test,selFeatures,how,seed):
	if how=='voting':
		clf1 = ExtraTreesClassifier(n_estimators=200, class_weight='balanced')
		clf2 = KNeighborsClassifier(n_neighbors=12)
		clf3 = SVC(gamma='scale', kernel='rbf', probability=True)       
		eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)], voting='soft', weights=[2,1,2])
		clf1 = clf1.fit(X_train.loc[:,selFeatures],y_train)
		clf2 = clf2.fit(X_train.loc[:,selFeatures],y_train)
		clf3 = clf3.fit(X_train.loc[:,selFeatures],y_train)
		eclf = eclf.fit(X_train.loc[:,selFeatures],y_train)
		y_pred=eclf.predict(X_test.loc[:,selFeatures])
	elif how=='xg':
		sumneg=sum(y_train==0)
		sumpos=sum(y_train==1)
		xgboost = XGBClassifier(scale_pos_weight=float(sumneg)/sumpos, missing=-1, n_estimators=200, random_state=seed, booster='gbtree',learning_rate=0.01,\
			max_depth=9, subsample=0.9, metric=xgb_f1)	
		xgboost.fit(X_train.loc[:,selFeatures], y_train)
		y_pred=xgboost.predict(X_test.loc[:,selFeatures])
	return y_pred

def featureSelection(method,X_train,y_train,trainingNoMismatch,seed,n_res):
	if method=='rl':
		#feature selection though stability selection
		#classToPredict could be 1 (sex) or 2 (msi)
		rl = RandomizedLasso(alpha='aic', random_state=seed, n_resampling=n_res)  #need high resampling to
		rl.fit(X_train,y_train) #predict sex 1 disease 2
		scores=rl.scores_
	elif method=='et':
		scores=np.zeros(X_train.shape[1])
		for i in range(n_res):
			xx_train, xx_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
			clf = ExtraTreesClassifier(n_estimators=100, class_weight='balanced')
			clf = clf.fit(xx_train, yy_train)
			scores=np.add(scores,clf.feature_importances_)
		scores=scores/n_res
	elif method=='xg':
		scores=np.zeros(X_train.shape[1])
		for i in range(n_res):
			xx_train, xx_test, yy_train, yy_test = train_test_split(X_train, y_train, test_size=0.25, random_state=i)
			sumneg=sum(yy_train==0)
			sumpos=sum(yy_train==1)
			xgboost = XGBClassifier(scale_pos_weight=float(sumneg)/sumpos, missing=-1, n_estimators=200, random_state=seed, booster='gbtree',learning_rate=0.01,\
				max_depth=9, subsample=0.9)        
			xgboost.fit(xx_train, yy_train)
			scores=np.add(scores,xgboost.feature_importances_)
		scores=scores/n_res

	names=list(trainingNoMismatch)
        #print "Features sorted by their score:"
        #print(sorted(zip(map(lambda x: round(x, 4), rl.scores_), names), reverse=True))
	vecSort=sorted(zip(map(lambda x: round(x, 4), scores), names), reverse=True)
	vecSortNum=[x[0] for x in vecSort]
	print('There are positive features', sum(x > 0 for x in vecSortNum))
	colNameVecSort=[x[1] for x in vecSort]

	return colNameVecSort

def fpreproc(dtrain, dtest, param):
    label = dtrain.get_label()
    ratio = float(np.sum(label == 0)) / np.sum(label==1)
    param['scale_pos_weight'] = ratio
    return (dtrain, dtest, param)

def xgb_f1(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
    return 'f1',f1_score(t,y_bin)

def xgb_f1_fmin(y,t):
    t = t.get_label()
    y_bin = [1. if y_cont > 0.5 else 0. for y_cont in y] # binaryzing your output
    return f1_score(t,y_bin)

def objective(params):
    num_round = int(params['n_estimators'])
    del params['n_estimators']
    watchlist = [(dmtrain, 'train'), (dmvalid, 'valid')]
    model = xgb.train(params, dmtrain, num_round, watchlist, maximize=True) #, early_stopping_rounds=20, verbose_eval=1)
    pred = model.predict(dmvalid, ntree_limit=model.best_ntree_limit)
    f1 = xgb_f1_fmin(pred, dmvalid)
    del model, pred
    gc.collect()
    print(f"SCORE: {f1}")
    return { 'loss': 1-f1, 'status': STATUS_OK }

###############
###Beginning###
###############
training,labels,matching,test,labelsTest=loadData()
labels,labelsTest=encodeData(labels,labelsTest)
impute='mean'
predictor='xg'
cvmethod='xg'
cv='grid'
mergeCols=0
if mergeCols==0:
	training,test=imputeMissing(impute,training,test)

#data exploration
training.max().plot(kind='hist')
training.isna().sum().plot(kind='hist')
sum(matching["mismatch"])

#remove mismatches for now
indices = np.where(matching["mismatch"]==0)[0]
misMatchInd = np.where(matching["mismatch"]==1)[0]
trainingNoMismatch = training.iloc[indices,:]
labelsNoMismatch = labels.iloc[indices,:]

if mergeCols==1:
	trainingNoMismatch.reset_index(drop=True, inplace=True)
	labelsNoMismatch.reset_index(drop=True, inplace=True)
	x = {'RPS4Y1': trainingNoMismatch.loc[:,"RPS4Y1"], 'RPS4Y2':trainingNoMismatch.loc[:,"RPS4Y2"],'EIF1AY': trainingNoMismatch.loc[:,'EIF1AY'], \
		'USP9Y': trainingNoMismatch.loc[:,'USP9Y'], 'DDX3Y':trainingNoMismatch.loc[:,"DDX3Y"], \
		'gender': labelsNoMismatch.loc[:,"gender"], 'msi': labelsNoMismatch.loc[:,"msi"]}
	df = pd.DataFrame(data=x)
	print(df.to_string())
	print(training.iloc[misMatchInd,:].loc[:,["RPS4Y1","RPS4Y2" ,"EIF1AY", "USP9Y", "DDX3Y"]].to_string())
	print(labels.iloc[misMatchInd,:].loc[:,"gender"])

for anteClass in [2,1]: #2 is predicting sex,1 is predicting msi
	if anteClass==1:
		classToPredict=2
	elif anteClass==2:
		classToPredict=1
	if cv=='manual':
		nCrossVal=100
		featVec=[1,2,3,4,5,6,7,8,9,10,15,16,17,18,19,20,30,40,50,60,70,80,90,100,training.shape[1]]
		scoreVec=np.zeros((nCrossVal,len(featVec)))
		for i in range(nCrossVal):
			#trainingNoMismatch['add1']=labelsNoMismatch.iloc[:,anteClass].values.astype(float)
			X_train, X_test, y_train, y_test = train_test_split(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict], test_size=0.2, random_state=i) #42
			n_res=10
			colNameVecSort=featureSelection(cvmethod,X_train,y_train,trainingNoMismatch,seed,n_res)
	
			#print(y_test)
			for j in range(len(featVec)):
				selFeatures=colNameVecSort[:featVec[j]] #select nFeatures
				y_pred=classifier(X_train,y_train,X_test,selFeatures,predictor,seed)
				a=f1_score(y_test,y_pred)
				scoreVec[i,j]=a
		scoreVec=np.mean(scoreVec, axis=0)
		print(list(zip(featVec,scoreVec)))
	elif cv=='automatic':
		#trainingNoMismatch['add1']=labelsNoMismatch.iloc[:,anteClass].values.astype(float)
		dtrain=xgb.DMatrix(trainingNoMismatch, label=labelsNoMismatch.iloc[:,classToPredict], missing=-1)
		param = {'sub_sample':0.9, 'silent':1, 'objective':'binary:logitraw','booster':'gbtree','num_round':1000} 
		cvResult=xgb.cv(param, dtrain, nfold=5, seed = 42+seed, fpreproc = fpreproc, feval= xgb_f1, metrics='aucpr')
		print(cvResult)
	elif cv=='grid':
		params = { # be careful uniform is lb+range // randint is lb,ub // I don't want to optimize lambda and alpha
        		'min_child_weight': sp_randint(1, 11),
        		'gamma': sp_uniform(0.5, 5.0-0.5),
        		'subsample': sp_uniform(0.6, 1.0-0.6),
        		'colsample_bytree': sp_uniform(0.6, 1.0-0.6),
        		'max_depth': sp_randint(1, 14),
			'n_estimators': sp_randint(600,3000),
        		'learning_rate': sp_uniform(0.001, 0.2-0.001),
			'objective': ['binary:logistic', 'binary:logitraw']}
		# Since startfiedkfold preserves class imbalance we can use balance classes
		sumneg=sum( labelsNoMismatch.iloc[:,classToPredict] == 0)
		sumpos=sum( labelsNoMismatch.iloc[:,classToPredict] == 1)
		xgb = XGBClassifier(silent=True, nthread=1, missing=-1, scale_pos_weight=float(sumneg)/sumpos) #add class imbalance
		folds = 5
		param_comb = 3000
		skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = seed)
		random_search = RandomizedSearchCV(xgb, param_distributions=params, n_iter=param_comb, scoring='f1', n_jobs=4, cv=skf.split(trainingNoMismatch,\
			labelsNoMismatch.iloc[:,classToPredict]), verbose=3, random_state=seed )
		random_search.fit(trainingNoMismatch, labelsNoMismatch.iloc[:,classToPredict])
		print('\n All results:')
		print(random_search.cv_results_)
		print('\n Best estimator:')
		print(random_search.best_estimator_)
		print('\n Best f1 score for %d-fold search with %d parameter combinations:' % (folds, param_comb))
		print(random_search.best_score_ )
		print('\n Best hyperparameters:')
		print(random_search.best_params_)
		random_search.best_params_["bestF1"]=random_search.best_score_
		results = pd.DataFrame(random_search.best_params_, index=[0])
		results.to_csv('xgb-random-grid-search-results' + str(classToPredict) +  '.csv')
		#save best model
		pickle.dump(random_search.best_score_, open('bestModel' + str(classToPredict), 'wb')) 
	elif cv=='fmin':
		# Create binary training and validation files for XGBoost
		# Not my favourite option because it does not cross-validate, and not sure if it ttok f1 as a metric, will come back
		dmtrain=xgb.DMatrix(trainingNoMismatch.iloc[:60,:], label=labelsNoMismatch.iloc[:60,classToPredict], missing=-1)
		dmvalid=xgb.DMatrix(trainingNoMismatch.iloc[60:,:], label=labelsNoMismatch.iloc[60:,classToPredict], missing=-1)
		space = {
    			# 'n_estimators': hp.quniform('n_estimators', 200, 600, 50),
    			'n_estimators': 3, # WARNING: increse number of estimators, e.g. uncomment the above line (it's small for the sake of example)
    			'eta': hp.quniform('eta', 0.025, 0.25, 0.025),
    			'max_depth': hp.choice('max_depth', np.arange(1, 14, dtype=int)),
    			'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
    			'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
    			'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
    			'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
    			'alpha' : hp.quniform('alpha', 0, 10, 1),
    			'lambda': hp.quniform('lambda', 1, 2, 0.1),
    			'scale_pos_weight': hp.quniform('scale_pos_weight', 50, 200, 10),
    			'objective': 'binary:logistic',
    			'feval': 'xgb_f1',
    			'tree_method': "hist",
    			'booster': 'gbtree',
    			'nthread': 4, 
    			'silent': 1
		}

		trials = Trials()
		best = fmin(
    			fn=objective,
    			space=space,
    			algo=tpe.suggest,
    			max_evals=10, # WARNING: increase number of evaluations (it's small for the sake of example)
    			trials=trials
			)

		# best hyperparameters
		print("\n\n\n The best hyperparameters:")
		print(best)


#To do:
#How to combine binary and continous features
#compute metric between pred and exp
#combine multipe metrics
#kappa cohen
#fill gender specific proteins by gender specific means

#More exploration for data imputation
#distribution of proteins by sex
#distribution of proteins by disease status
#impute median/mean/sample between mean and sd
#impute by row mean or col mean
#scale + standardize + fill missing
#Optimize hyperparameters
#new idea to find x chr prot less than 50 missing

#Problems:
#not same set of features selected-> ET classifer
#Why such low mean for UDP9Y as in 1e-15?-> scaling
#class balance in randomized lasso
#missing in xgboost

#Note:
#gblinear is better with msiand gbtree with sex
#seed=42 does not improve with gradient boost, better with imputation to 0
#try PCA
#optimize xgboost grid searcch param
#look for parameter tha optimizes tp
#look at john's paper

#xgboost hyperparameters
#Try fmin
#try actual random in ranges
#laod balance
