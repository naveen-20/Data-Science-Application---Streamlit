import streamlit as st 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
#from sklearn.decompostion import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Title
st.title('Advance Data Science Application Using Streamlit')
st.subheader("Let's explore data" )

#Creating SideBar
dataset_name = st.sidebar.selectbox("Select Dataset",('Breast Cancer','Iris','Wine'))
classifier_name = st.sidebar.selectbox("Select Classification Algorithmn",('SVM','KNN'))

#Reading dataset
#if dataset_name == 'Iris':
#	st.dataframe(datasets.load_iris().data)
#elif dataset_name == 'Wine':
#	st.dataframe(datasets.load_wine().data)
#elif dataset_name == 'Breast Cancer':
#	st.dataframe(datasets.load_breast_cancer().data)

#Reading dataset with function

def get_data(name):
	if dataset_name == 'Iris':
		data=datasets.load_iris()
	elif dataset_name=='Wine':
		data=datasets.load_wine()
	else:
		data=datasets.load_breast_cancer()
	df = pd.DataFrame(data.data, columns=data.feature_names)
	df['Target'] = data.target
	x=df
	y=df['Target']

	return x,y
x,y = get_data(dataset_name)
st.dataframe(x)
st.write("Dataset Shape : ",x.shape)
st.write("Unique Targets : ", len(np.unique(y)))

#Visualization
st.subheader("Data Distribution")
st.text('Box Plot')
st.set_option('deprecation.showPyplotGlobalUse', False)
fig=plt.figure(figsize=(12,8))
sns.boxplot(data=x, orient='h')
st.pyplot()

#Histogram
feature=st.selectbox('Select a feature to plot histogram',(x.columns))
sns.histplot(data=x[feature])
st.pyplot()

#Correlation Chart
st.subheader('Data Correlation')
st.text('Heat Map')
fig=plt.figure(figsize=(20,10))
sns.heatmap(x.corr(),annot=True)
st.pyplot()

#Algorithm Building
#selecting parameter values.

def get_parameter(algo):
	param = dict()
	if algo == 'SVM':
		c=st.sidebar.slider('C',1,15)
		param['C'] = c
	else:
		k=st.sidebar.slider('K',1,10)
		param['K']=k
	return param

param=get_parameter(classifier_name)

#intializing algorithm
def get_algo(algo_name, param):
	if algo_name=='SVM':
		clf = SVC(C=param['C'])
	else:
		clf = KNeighborsClassifier(n_neighbors=param['K'])
	return clf

clf= get_algo(classifier_name,param)

#train test split
X_train, X_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=0)
clf.fit(X_train, y_train)
predicted_values = clf.predict(X_test)
accuracy = accuracy_score(predicted_values,y_test)
st.subheader('Machine Learning Model Prediction')
st.write('Classifier name : ', classifier_name)
st.write('Model Accuracy is:' ,accuracy)






