# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:11:55 2024

@author: TOM
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pickle

#--------------------
# cleaning data base 
#--------------------

df = pd.read_csv('\\Python projects\\train_u6lujuX_CVtuZ9i.csv')

 
df.info()
#visualisation du nombre de données manquantes par catégories
df.isnull().sum().sort_values(ascending = False)

df.describe(include='O')

#on cherche à séparer les valeurs numériques des valeurs catégoriques (cat = catégorique / num = numérique)
cat_data=[]
num_data=[]

for i,c in enumerate(df.dtypes):
    if c==object:
        cat_data.append(df.iloc[:,i])
    else :
        num_data.append(df.iloc[:,i])
        
cat_data =pd.DataFrame(cat_data).transpose()
num_data =pd.DataFrame(num_data).transpose()

#remplacer les variables manquantes par les variables catégoriques qui se répètent le plus

cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))

#remplacer les variables manquantes par les variables numériques par la valeur précédente de la même colonne
num_data.fillna(method = 'bfill', inplace=True)
num_data.isnull().sum().any()

#Transformer la colonne "Loan_Status" par que des 1 ou 0 (façon manuelle)
target_value={'Y':1, 'N':0}
target=cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)
target=target.map(target_value)

#Transformer toutes les colonnes par des 0,1,2... (façon raccourcie)
le=LabelEncoder
for i in cat_data:
    cat_data[i]= le.fit_transform(cat_data[i])
    
#Supprimer loan_id car elle ne sert pas à grand chose
cat_data.drop('Loan_ID',axis=1,inplace=True)

#concaténer cat_data et num_data

X=pd.concat([cat_data,num_data],axis=1)
Y=target

#%%

#-------------------------------
#Exploratory Data Analysis (EAD)
#-------------------------------

target.value_counts()

plt.figure(figsize=(8,6))

sns.countplot(target)

yes= target.value_counts()[0]/len(target)
no=target.value_counts()[1]/len(target)

print(f'{yes} de crédit accordé')
print(f'{no} de crédit refusé')

df=pd.concat([cat_data,num_data,target],axis=1)

#etude de la relation entre l'acceptation du crédit et l'historique de crédit
grid=sns.FaceGrid(df,col='Loan_Status',size=3.2,aspect=1.6)
grid.map(sns.countplot,'Credit_History')

#etude de la relation entrel'acceptation du crédit et le sexe de la personne
grid=sns.FaceGrid(df,col='Loan_Status',size=3.2,aspect=1.6)
grid.map(sns.countplot,'Gender')

#etude de la relation entrel'acceptation du crédit et la situation familiale 
grid=sns.FaceGrid(df,col='Loan_Status',size=3.2,aspect=1.6)
grid.map(sns.countplot,'Married')

#etude par rapport au revenu
plt.scatter(df['ApplicantIncome'],df['Loan_Status'])

#etude par rapport du revenu du conjoint
plt.scatter(df['CoapplicantIncome'],df['Loan_Status'])

df.groupby('Loan_Status').median()

#%%

#-----------------
#Model realisation
#-----------------

# diviser la base de données en une base de données test et d'entrainement 

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train, test in sss.split(X,Y):
    X_train, X_test=X.iloc[train],X.iloc[test]
    Y_train, Y_test=Y.iloc[train],Y.iloc[test]
    
# 3 algo: Logistic Regression, KNN, Decision Tree

models={
       'logisticRegression ':LogisticRegression(random_state=42) 
       'KNeighborsClassifier':KNeighborsClassifier() 
       'DecisionTreeClassifier':DecisionTreeClassifier(max_depth=1, random_state=42)
       
       }

#la fonction de precision
def accu(Y_true,Y_pred,retu=False):
    acc = accuracy_score(Y_true,Y_pred)
    if retu:
        return acc
    else:
        print(f'la precision du modèle est :{acc}')
        
#la fonction d'application des modèles
def train_test_eval(models,X_train,Y_train,X_test,Y_test):
    for name,model in models.items():
        print(name,':')
        model.fit(X_train,Y_train)
        accu(Y_test, model.predict(X_test))
        print('-'*30)

train_test_eval(models,X_train,Y_train,X_test,Y_test)
        
# choix de quelques variables qu'on donnera en entré à notre modèle, cela va éviter de devoir donner les 11 variables

X_2=X[['Credit_History','Married','CoapplicantIncome']]

sss = StratifiedShuffleSplit(n_splits=1,test_size=0.2, random_state=42)
for train, test in sss.split(X_2,Y):
    X_train, X_test=X_2.iloc[train],X_2.iloc[test]
    Y_train, Y_test=Y.iloc[train],Y.iloc[test]

train_test_eval(models,X_train,Y_train,X_test,Y_test)

#%%
# On choisi le modèle le plus performant (Regression Logistic)

#appliquer la regression logistique sur notre BDD
Classifier=LogisticRegression()
Classifier.fit(X_2,Y)

#Enregistrer le modèle dans un dossier 

pickle.dump(Classifier,open('model.pkl',"wb"))
