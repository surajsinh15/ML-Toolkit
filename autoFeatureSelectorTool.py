import numpy as np
import pandas as pd 
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler

def cor_selector(X, y, num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature

def chi_squared_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()
    return chi_support, chi_feature

def rfe_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    rfe_selector.fit(X_norm, y)
    rfe_support = rfe_selector.get_support()
    rfe_feature = X.loc[:, rfe_support].columns.tolist()
    return rfe_support, rfe_feature

def embedded_log_reg_selector(X, y, num_feats):
    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l2"), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()
    return embedded_lr_support, embedded_lr_feature

def embedded_rf_selector(X, y, num_feats):
    embedded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)
    embedded_rf_selector.fit(X, y)
    embedded_rf_support = embedded_rf_selector.get_support()
    embedded_rf_feature = X.loc[:, embedded_rf_support].columns.tolist()
    return embedded_rf_support, embedded_rf_feature

def embedded_lgbm_selector(X, y, num_feats):
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2,
                          reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embedded_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embedded_lgbm_selector.fit(X, y)
    embedded_lgbm_support = embedded_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()
    return embedded_lgbm_support, embedded_lgbm_feature

def preprocess_dataset(dataset_path):
        player_df = pd.read_csv(dataset_path)
        numcols = ['Overall', 'Crossing','Finishing',  'ShortPassing',  'Dribbling','LongPassing', 'BallControl', 'Acceleration','SprintSpeed', 'Agility',  'Stamina','Volleys','FKAccuracy','Reactions','Balance','ShotPower','Strength','LongShots','Aggression','Interceptions']
        catcols = ['Preferred Foot','Position','Body Type','Nationality','Weak Foot']
        player_df = player_df[numcols+catcols]
        traindf = pd.concat([player_df[numcols], pd.get_dummies(player_df[catcols])],axis=1)
        features = traindf.columns
        traindf = traindf.dropna()
        traindf = pd.DataFrame(traindf,columns=features)
        y = traindf['Overall']>=87
        X = traindf.copy()
        del X['Overall']
        num_feats=30
        return X, y, num_feats

def autoFeatureSelector(dataset_path, methods=[]):
        
        X, y, num_feats = preprocess_dataset(dataset_path)
      
        selected_features = {}
  
    
        if 'pearson' in methods:
            cor_support, cor_feature = cor_selector(X, y, num_feats)
            selected_features['pearson'] = cor_feature
        if 'chi-square' in methods:
            chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
            selected_features['chi-square'] = chi_feature
        if 'rfe' in methods:
            rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
            selected_features['rfe'] = rfe_feature
        if 'log-reg' in methods:
            embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
            selected_features['log-reg'] = embedded_lr_feature
        if 'rf' in methods:
            embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
            selected_features['rf'] = embedded_rf_feature
        if 'lgbm' in methods:
            embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)
            selected_features['lgbm'] = embedded_lgbm_feature
       
        common_features = set(selected_features['pearson'])  
        for method, features in selected_features.items():
            common_features.intersection_update(features)
   
        best_features = list(common_features)
       
        return best_features
     
dataset_path = input("Enter the dataset file path:" )
    methods = input("Enter a comma-separated list of feature selection methods (e.g., pearson,chi-square,rfe):" ).split(',')
   
    methods = [method.strip() for method in methods]
  
    best_features = autoFeatureSelector(dataset_path=dataset_path, methods=methods)
  
    print("Best features:", best_features)