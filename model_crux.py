import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

np.random.seed(42)
data = pd.read_csv('data/heart.csv')

X = data.drop('target',axis=1)
Y = data.target

x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

clf = LogisticRegression(C=0.23357214690901212,solver='liblinear')
clf.fit(x_train,y_train)

print(" Score :",clf.score(x_test,y_test)*100,"%")
pickle.dump(clf,open('final_model.pkl','wb'))