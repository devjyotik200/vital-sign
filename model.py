# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle


dataset=pd.read_csv("cardio_train.csv",sep=';')


correlation=dataset.corr()

y=dataset.iloc[:,12].values

dataset=dataset.drop(['id','gender','height','weight'],axis=1)

x=dataset.iloc[:,:].values

x[:,0]=x[:,0]/365     #CONVERTING DAYS TO YEARS
x=x[:,0:-1]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train,y_test=train_test_split(x,y,test_size=0.10,random_state=0)

#TO PREDICT WHETHER THE PERSON HAS ANY CARDIOVASCULAR DISEASE



from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(max_depth=10,random_state=101,max_features=None,min_samples_leaf=15)
classifier.fit(x_train,y_train)

pickle.dump(classifier, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))