from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from collections import OrderedDict
from catboost import CatBoostRegressor, CatBoostClassifier
#import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.metrics import recall_score, accuracy_score, balanced_accuracy_score, precision_score, confusion_matrix, log_loss, roc_auc_score, brier_score_loss
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV, Ridge, RidgeCV, ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold, cross_val_predict,  cross_val_score
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import RidgeClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import time
import pickle

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# pip install catboost lightgbm
CLASSMODELS= OrderedDict()
CLASSMODELS['LogsReg'] = LogisticRegression()
# CLASSMODELS['Ridge-Alpha-0.5']  = RidgeClassifier(alpha=0.50)
# CLASSMODELS['Ridge-Alpha-1.0']  = RidgeClassifier(alpha=1.0)
#CLASSMODELS['Ridge-Alpha-5.0']  = RidgeClassifier(alpha=5.0)
#CLASSMODELS['Ridge-Alpha-10.0']  = RidgeClassifier(alpha=10.0)
CLASSMODELS['CatBoostClassifier'] = CatBoostClassifier(verbose=False)
CLASSMODELS['GradBoostClass'] = GradientBoostingClassifier() #Takes the longest
# CLASSMODELS['LightGBM']  = lgb.LGBMClassifier()
# CLASSMODELS['Decision-Tree'] = DecisionTreeClassifier() #Commented for time
# CLASSMODELS['GaussianNB'] = GaussianNB()

CLASSMODELS['XGB'] = xgb.XGBClassifier()
#CLASSMODELS['RandomForestClassifier'] = RandomForestClassifier()
# CLASSMODELS['AdaBoostClassifier'] = AdaBoostClassifier()

# CLASSMODELS['SGDClassifier'] = SGDClassifier(loss='log', penalty='elasticnet', alpha=0.0001)


REGMODELS = OrderedDict()
REGMODELS['LinearRegression']=LinearRegression()
REGMODELS['RidgeRegression']=Ridge()
#REGMODELS['Lasso-Alpha-5.0']=Lasso(alpha=5)
#REGMODELS['Lasso-Alpha-10.0']=Lasso(alpha=10)
REGMODELS['CatBoostRegressor']=CatBoostRegressor(logging_level='Silent')
REGMODELS['GradientBoostingRegression']=GradientBoostingRegressor()
REGMODELS['XGBoostRegressor']=xgb.XGBRegressor()
#REGMODELS['RandomForrestRegressor']=RandomForestRegressor(max_depth = 6) Too long even with thais.



def get_regmodels(models=REGMODELS):
    return models.keys()

def get_classmodels(models=CLASSMODELS):
    return models.keys()

def score_regression(experiment, dv, model, testfold,  y_true, y_pred, time):
    """
    This function evaluates a binary classifier and returns a Pandas Series of metrics.

    Parameters:
    experiment : str
    dv : str
    model : sklearn model
    over_strategy : str
    over_algo : str
    over_ratio : str
    testfold : str
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    y_prob : array-like of shape (n_samples,)

    Returns:
    result : pd.Series
        A series with evaluation metrics.
    """

    # Initialize result as a Pandas Series
    result = pd.Series(dtype='object')

    # Assign values to result
    result['sysdate'] = pd.Timestamp.now(tz=None)
    result['sample_size'] = len(y_pred)
    result['experiment'] = experiment
    result['train_time']= time
    result['dv'] = dv
    result['model'] = str(model)
    result['train_test'] = testfold
    result['r2'] =r2_score(y_true, y_pred)  
    result['mse'] =mean_squared_error(y_true, y_pred)
    result['rmse'] = mean_squared_error(y_true, y_pred, squared=False)
    result['mae'] =mean_absolute_error(y_true, y_pred)
    result=result.to_frame().transpose()
    return result

def score_binary(experiment, dv, model, testfold,  y_true, y_pred, y_prob):
    """
    This function evaluates a binary classifier and returns a Pandas Series of metrics.

    Parameters:
    experiment : str
    dv : str
    model : sklearn model
    over_strategy : str
    over_algo : str
    over_ratio : str
    testfold : str
    y_true : array-like of shape (n_samples,)
    y_pred : array-like of shape (n_samples,)
    y_prob : array-like of shape (n_samples,)

    Returns:
    result : pd.Series
        A series with evaluation metrics.
    """

    # Check inputs
    if len(y_true) != len(y_pred) or len(y_true) != len(y_prob):
        raise ValueError("Input arrays y_true, y_pred, and y_prob must have the same length")

    # Initialize result as a Pandas Series
    result = pd.Series(dtype='object')

    # Assign values to result
    result['sysdate'] = pd.Timestamp.now(tz=None)
    result['sample_size'] = len(y_pred)
    result['experiment'] = experiment
    result['dv'] = dv
    result['model'] = str(model)
    result['train_test'] = testfold
    #result['acc_scores'] = acc_scores
    #result['acc_score_mean']=np.mean(result['acc_scores'])
    #result['roc_auc_scores'] = roc_auc_scores
    #result['roc_auc_mean']=np.mean(result['roc_auc_scores'])

    result['recall'] = recall_score(y_true, y_pred, average='binary')
    result['accuracy'] = accuracy_score(y_true, y_pred)
    result['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)
    result['precision'] = precision_score(y_true, y_pred, average='binary')
    result['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    result['log_loss'] = log_loss(y_true, y_prob)
    result['roc_auc_score'] = roc_auc_score(y_true, y_prob)
    result['brier_score_loss'] = brier_score_loss(y_true, y_prob)
    result=result.to_frame().transpose()
    return result

def train_test_split_validation(dict_input, return_test=False):
    if dict_input['type']=='regression':
        allmodels=REGMODELS
    elif dict_input['type']=='classification':
        allmodels=CLASSMODELS

    start_time = time.time()
    # Extract information from the input dictionary
    results= pd.DataFrame()
    experiment = dict_input['experiment']
    df = dict_input['df']
    categorical_cols = dict_input['cat']
    numerical_cols = dict_input['num']
    dependent_var = dict_input['dv']
    model = allmodels[dict_input['model']]
    training_weeks = dict_input['training_weeks']
   # df.set_index(dict_input['key'][0], inplace=True)


    # Create preprocessing pipelines for both numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)])

    # Append classifier to preprocessing pipeline
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    

    train = df[df["week_num"] <= training_weeks]
    if 'test_weeks' in dict_input:
        test = df[(df['week_num'] > 55) & (df['week_num'] <= (55 + dict_input['test_weeks']))]
    else:
        test = df[(df['week_num'] > 55)]
    # Split the dataset into training and testing sets
    all_X=numerical_cols+categorical_cols
   
    X_train=train[ all_X] 
    X_test=test[ all_X]
  
    y_train = train[dependent_var]
    y_test = test[dependent_var]

    # Fit the classifier on the training data
    clf.fit(X_train, y_train)

    # Predict on the test data
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    end_time = time.time()
    total_time = end_time - start_time
    # Calculate metrics using the score_binary function
    test[dependent_var+'-'+dict_input['model']]=y_pred_test
    if dict_input['type']=='regression':
        metrics_train = score_regression(experiment, dependent_var, dict_input['model'], 'train', y_train, y_pred_train, total_time)
        metrics_test = score_regression(experiment, dependent_var, dict_input['model'], 'test', y_test, y_pred_test, total_time)
        results=pd.concat([results, metrics_train, metrics_test])
        test=test[['auction_id',dependent_var,dependent_var+'-'+dict_input['model']]]

    elif dict_input['type']=='classification':
        y_prob_train = clf.predict_proba(X_train)[:, 1]
        y_prob_test = clf.predict_proba(X_test)[:, 1]
        test[dependent_var+'-'+dict_input['model']+'-prob']=y_prob_test
        metrics_train = score_binary(experiment, dependent_var, dict_input['model'], 'train', y_train, y_pred_train, y_prob_train)
        metrics_test = score_binary(experiment, dependent_var, dict_input['model'], 'test', y_test, y_pred_test, y_prob_test)
        results=pd.concat([results, metrics_train, metrics_test])
        test=test[['auction_id',dependent_var,dependent_var+'-'+dict_input['model'],dependent_var+'-'+dict_input['model']+'-prob']]
    if return_test:
        return results, test
    else: 
        return results
    

