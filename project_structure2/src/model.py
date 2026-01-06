# src/model.py
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def get_logistic_model(C=0.04, max_iter=400, penalty='l1', solver='liblinear'):
    return LogisticRegression(C=C, max_iter=max_iter, penalty=penalty, solver=solver)

def get_linear_svc(C=1.0, max_iter=200, penalty='l2', fit_intercept=False):
    return LinearSVC(C=C, max_iter=max_iter, penalty=penalty, fit_intercept=fit_intercept)

def get_decision_tree(criterion='gini', max_depth=200, max_leaf_nodes=500, splitter='best'):
    return DecisionTreeClassifier(criterion=criterion, max_depth=max_depth,
                                  max_leaf_nodes=max_leaf_nodes, splitter=splitter)

def get_random_forest(n_estimators=100, max_depth=20, criterion='gini'):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                  criterion=criterion)

def get_xgb_classifier(n_estimators=40, max_depth=20, learning_rate=0.2, gamma=0.002, reg_alpha=0.1):
    return xgb.XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                             learning_rate=learning_rate, gamma=gamma, reg_alpha=reg_alpha)
