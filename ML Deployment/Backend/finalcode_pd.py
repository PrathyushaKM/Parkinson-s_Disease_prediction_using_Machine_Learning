import numpy as np
import pandas as pd
import pickle
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
#import os, sys
from sklearn.preprocessing import MinMaxScaler #to scale features
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix,precision_score,f1_score,classification_report
import seaborn as sns
sns.set(color_codes=True) # adds a background to the graphs
import warnings
warnings.filterwarnings('ignore')
print('all packages are imported')


# In[7]:


import os
print(os.getcwd())


# In[8]:


data=pd.read_csv('123.csv')
data.head()


# In[23]:


data.columns


# In[24]:


data.dtypes


# In[25]:


data[data.duplicated()]


# In[26]:


print('informations about the dataset')
data.info()
# ensure that there are no Null Values
print("Null Values Check\n")
print(data.isnull().sum())
print("\n\n NAN Values Check \n")
print(data.isna().sum())


# In[28]:


sns.countplot(data['status'])

# Add labels
plt.title('Countplot of status')
plt.xlabel('status')
plt.ylabel('Patients')
plt.show()


# In[29]:


data.describe().transpose()


# In[30]:


print('The measures of vocal fundamental frequency')
fig, ax = plt.subplots(1,3,figsize=(15,6)) 
sns.distplot(data['MDVP:Flo(Hz)'],ax=ax[0]) 
sns.distplot(data['MDVP:Fo(Hz)'],ax=ax[1]) 
sns.distplot(data['MDVP:Fhi(Hz)'],ax=ax[2])


# In[31]:


def distributionPlot(data):
    fig, ax = plt.subplots(2,3,figsize=(16,8)) 
    sns.distplot(data['MDVP:Shimmer'],ax=ax[0,0],color='red') 
    sns.distplot(data['MDVP:Shimmer(dB)'],ax=ax[0,1],color='gray') 
    sns.distplot(data['Shimmer:APQ3'],ax=ax[0,2],color='pink') 
    sns.distplot(data['Shimmer:APQ5'],ax=ax[1,0],color='blue') 
    sns.distplot(data['MDVP:APQ'],ax=ax[1,1],color='green') 
    sns.distplot(data['Shimmer:DDA'],ax=ax[1,2],color='Aquamarine')
    plt.show()
distributionPlot(data)


# In[32]:


fig, ax = plt.subplots(1,2,figsize=(10,7)) 
sns.distplot(data['NHR'],ax=ax[0],color='cyan') 
sns.distplot(data['HNR'],ax=ax[1],color='pink')


# In[33]:


fig, ax = plt.subplots(1,3,figsize=(10,7)) 
sns.boxplot(y='spread1',data=data, ax=ax[0],orient='v') 
sns.boxplot(y='spread2',data=data, ax=ax[1],orient='v')
sns.boxplot(y='PPE',data=data,ax=ax[2],orient='v')


# In[5]:


fig,axes=plt.subplots(5,5,figsize=(15,15))
axes=axes.flatten()

for i in range(1,len(data.columns)-1):
    sns.boxplot(x='status',y=data.iloc[:,i],data=data,orient='v',ax=axes[i],palette = 'copper')
plt.tight_layout()
plt.show()


# In[35]:


#Get the features and labels from the DataFrame ( 23 features and 1 label (statut))
features= data.drop(['status','name'], axis = 1)
#features=data.loc[:,data.columns!='status','name'].values[:,1:] #Selecting data by label or by a conditional statement [<row selection>,<column selection>]
labels=data.loc[:,'status'].values
#features=data.drop(['status','name'],axis=1)
#labels=data['status']


# In[36]:


#Get the count of each label (0 and 1) in labels
print('number of parkinson people in the dataset:',labels[labels==1].shape[0])
print('number of Healthy people in the dataset: ',labels[labels==0].shape[0]) #The shape attribute for numpy arrays returns the dimensions of the array
#1 : 147 , 0 : 48


# In[37]:


#Scale the features to between -1 and 1 (normalisation)
scaler=MinMaxScaler((-1,1))
x=scaler.fit_transform(features)
y=labels


# In[38]:


#split the dataset into training and testing sets keeping 20% of the data for testing.
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.2, random_state=2)
print("{0:.1f}% data is in training set".format((len(x_train)/len(data)) * 100))
print("{0:.1f}% data is in testing set".format((len(x_test)/len(data)) * 100))
#overview of data
print(x_train.shape)
print(x_test.shape)


# In[4]:


from sklearn.neighbors import KNeighborsClassifier
#determine the optimum value of k 
train_score = []
test_score = []
k_vals = []

for k in range(1, 21):
    k_vals.append(k)
    mod_knn = KNeighborsClassifier(n_neighbors = k)
    mod_knn.fit(x_train, y_train)
    
    tr_score = mod_knn.score(x_train, y_train)
    train_score.append(tr_score)
    
    te_score = mod_knn.score(x_test, y_test)
    test_score.append(te_score)

# score that comes from the testing set only
 
max_test_score = max(test_score)
test_scores_ind = [i for i, v in enumerate(test_score) if v == max_test_score]
print('Max test score {} and k = {}'.format(max_test_score * 100, list(map(lambda n: n + 1, test_scores_ind))))
 


# In[2]:


model_knn = KNeighborsClassifier(11)

model_knn.fit(x_train, y_train)
model_knn.score(x_test, y_test)*100


# In[1]:


pred_knn=model_knn.predict(x_test)
 
#accuracy metrix
test_acc=accuracy_score(y_test,pred_knn)*100
print('the accuracy of the model',test_acc)

c_matrix = confusion_matrix(y_test,pred_knn)
print('Confusion matrix : \n',c_matrix)

# outcome values order in sklearn
tn, fp, fn, tp = confusion_matrix(y_test,pred_knn).reshape(-1)
print('Outcome values : \n', tn, fp, fn, tp)

# classification report for precision, recall f1-score and accuracy
cl_rep = classification_report(y_test,pred_knn)
print('Classification report : \n',cl_rep)


# In[42]:


sns.heatmap(c_matrix, annot= True, cmap='Blues')
plt.title('The confusion matrix')
plt.xlabel('predicted values')
plt.ylabel('actual values')
plt.show()

with open('finalcode_pd.pkl','wb') as f:
    pickle.dump(model_knn,f)
 
model=pickle.load(open('finalcode_pd.pkl','rb'))
print(model) 


