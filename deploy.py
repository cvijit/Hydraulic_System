import streamlit as st 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression 
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import bokeh
import bokeh.layouts
import bokeh.models
from bokeh.models.widgets import Div


#Link to the dashboard in the web app
if st.button('EDA Dashboard'):
    js = "window.open('https://hydrauliceda.shinyapps.io/draft1_blank/#section-data-exploration/')"  # New tab or window
    js = "window.location.href = 'https://hydrauliceda.shinyapps.io/draft1_blank/#section-data-exploration/'"  # Current tab
    html = '<img src onerror="{}">'.format(js)
    div = Div(text=html)
    st.bokeh_chart(div)

#title 
st.title('Research Project')

st.write("""
# Explore different classifier and datasets to monitor status of Hydraulic System and their components

""")

#Choosing the dataset
dataset_name = st.sidebar.selectbox(
    'Select Dataset',
    ('Coolers', 'Valves', 'Pump_Leakage', 'Hydraulic_Accumulator','Stable_Condition')
)

st.write(f"## {dataset_name} Dataset")

#Choosing the classification algorithm
classifier_name = st.sidebar.selectbox('Select classifier', 
                                      ('Logistic Regression','XGBoost', 'Light GBM',
                                       'Catboost','Random Forest')
)

#Reading the csv foles from GitHub
url_1 = 'https://github.com/cvijit/hydraulics-research/blob/master/cooler_deploy.csv?raw=true'
url_2 = 'https://github.com/cvijit/hydraulics-research/blob/master/valve_deploy.csv?raw=true'
url_3 = 'https://github.com/cvijit/hydraulics-research/blob/master/pump_deploy.csv?raw=true'
url_4 = 'https://github.com/cvijit/hydraulics-research/blob/master/accumulator_deploy.csv?raw=true'
url_5 = 'https://github.com/cvijit/hydraulics-research/blob/master/stable_deploy.csv?raw=true'

#Function to read the dataset according to the choosen dataset
def get_dataset(dataset_name):
    if dataset_name == 'Coolers':
        data = pd.read_csv(url_1,index_col=0)
    elif dataset_name == 'Valves':
        data = pd.read_csv(url_2,index_col=0)
    elif dataset_name == 'Pump_Leakage':
        data = pd.read_csv(url_3,index_col=0)
    elif dataset_name == 'Hydraulic_Accumulator':
        data = pd.read_csv(url_4,index_col=0)
    else:
        data = pd.read_csv(url_5,index_col=0)
    X = data.iloc[:,0:17]
    y= data.iloc[:,-1]
    return X, y

X, y= get_dataset(dataset_name)

#Defining the shape and number of the dataset
st.write('Shape of dataset', X.shape)
st.write('Number of classes',len(np.unique(y)))


#Selecting parameters for classification algorithm
def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "Logistic Regression":
        C = st.sidebar.slider("C",0.001,0.1)
        params["C"] = C
        max_iter = st.sidebar.slider("max_iter",500,3000)
        params["max_iter"]= max_iter
        solver = st.sidebar.selectbox("solver",('lbfgs','newton-cg','liblinear','sag','saga'))
        params["solver"]=solver
    elif clf_name == "XGBoost":
        min_child_weight = st.sidebar.slider("min_child_weight ", 0,5)
        params['min_child_weight']=min_child_weight
        max_depth = st.sidebar.slider("max_depth " ,1,10)
        params['max_depth']=max_depth
        learning_rate = st.sidebar.slider("learning_rate ", 0.01,1.0)
        params['learning_rate']=learning_rate
        gamma = st.sidebar.slider("gamma ", 0.01,1.0)
        params['gamma']=gamma
        colsample_bytree = st.sidebar.slider("colsample_bytree ", 0,1)
        params['colsample_bytree']=colsample_bytree
    elif clf_name == "Light GBM":
        objective = st.sidebar.selectbox("objective",('binary','multiclass'))
        params["objective"]=objective
        metric = st.sidebar.selectbox("metric",('multi_logloss','roc','auc'))
        params["metric"]=metric  
        boosting_type = st.sidebar.selectbox("boosting_type",('gbdt','dart','goss'))
        params["boosting_type"]=boosting_type
        learning_rate = st.sidebar.slider("learning_rate ", 0.01,1.0)
        params['learning_rate']=learning_rate
        max_depth = st.sidebar.slider("max_depth " ,1,10)
        params['max_depth']=max_depth
        colsample_bytree = st.sidebar.slider("colsample_bytree ", 0,1)
        params['colsample_bytree']=colsample_bytree                 
    elif clf_name == "CatBoost":
        bootstrap = st.sidebar.selectbox("bootstrap",('True','False'))
        params['bootstrap'] = bootstrap
        criterion = st.sidebar.selectbox("criterion",('entropy','gini'))
        params['criterion'] = criterion
        max_depth = st.sidebar.slider("max_depth",1,20)
        params['max_depth'] = max_depth
        max_features = st.sidebar.slider("max_features",0.1,0.2)
        params['max_features'] = max_features
        n_estimators = st.sidebar.slider("n_estimators",100,500)
        params['n_estimators'] = n_estimators
    else:
        bootstrap = st.sidebar.selectbox("bootstrap",('True','False'))
        params['bootstrap'] = bootstrap
        criterion = st.sidebar.selectbox("criterion",('entropy','gini'))
        params['criterion'] = criterion
        max_depth = st.sidebar.slider("max_depth",1,20)
        params['max_depth'] = max_depth
        max_features = st.sidebar.slider("max_features",0.1,0.2)
        params['max_features'] = max_features
        n_estimators = st.sidebar.slider("n_estimators",100,500)
        params['n_estimators'] = n_estimators
        min_samples_leaf = st.sidebar.slider("min_samples_leaf",1,4)
        params['min_samples_leaf'] = min_samples_leaf
    return params

params = add_parameter_ui(classifier_name)

#Fitting the parameters into the choosen classification model
def get_classifier(clf_name, params):
    if clf_name == 'Logistic Regression':
        clf = LogisticRegression(C=params["C"], max_iter=params["max_iter"],
                                 solver=params["solver"])
    elif clf_name == 'XGBoost':
        clf = xgb.sklearn.XGBClassifier(min_child_weight=params['min_child_weight'], 
                                        max_depth=params['max_depth'], learning_rate=params['learning_rate'], 
                                        gamma=params['gamma'], colsample_bytree=params['colsample_bytree'])
    elif clf_name == 'Light GBM':
        clf = lgb.LGBMClassifier(objective=params["objective"],metric=params["metric"],
                                 boosting_type=params["boosting_type"],learning_rate=params["learning_rate"],
                                 max_depth=params['max_depth'],colsample_bytree=params['colsample_bytree'])
    elif clf_name == 'CatBoost':
        clf = CatBoostClassifier(bootstrap=params['bootstrap'], criterion=params['criterion'],
                                     max_depth=params['max_depth'],max_features=params['max_features'],
                                     n_estimators=params['n_estimators'])
    else:
        clf = RandomForestClassifier(bootstrap=params['bootstrap'], criterion=params['criterion'],
                                     max_depth=params['max_depth'],max_features=params['max_features'],
                                     n_estimators=params['n_estimators'],min_samples_leaf=params['min_samples_leaf'])
    return clf

clf = get_classifier(classifier_name, params)


#### CLASSIFICATION ####
#Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
#Fitting the model
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
#Evaluating the model
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred) 
#Displaying Results
st.write(f'Classifier = {classifier_name}')
st.write(f'Accuracy =', acc*100)
st.write(f'Confusion matrix =', cm)










