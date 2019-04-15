import pandas as pd
import numpy as np
from sklearn.metrics import f1_score #For f1 score

result      = pd.read_csv("../data/results/sub1_key.csv", sep=",")
scores=[]
for i in range(1,11):
	prediction  = pd.read_csv("../data/predictions/"+ str(i) + "/submission" + str(i) + ".csv", sep=",")
	score=f1_score(result['mismatch'], prediction['mismatch'])
	scores.append(score)

print(np.mean(scores))