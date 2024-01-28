"""
Created on Jan 27, 2024.
main_RADADA.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import os
import numpy as np
from sklearn import metrics
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score
from scipy.stats import norm
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import warnings
warnings.filterwarnings('ignore')



class Validation():
    def __int__(self):
        pass

    def main_train_GPT_ADA(self):

        # Load the training dataset
        train_file_path = '/mnt/data/full_task4_train_set_CXR_LabValues_Aachen.xlsx'
        train_df = pd.read_excel(train_file_path)

        # Display the first few rows of the dataset to understand its structure
        train_df.head()

        # Selecting features and target variable
        X = train_df.drop(['patient_ID', 'Exam_Date', 'Exam_Time', 'Pulmonary_Opacities', 'split'], axis=1)
        y = train_df['Pulmonary_Opacities']

        # Identifying numerical and categorical columns
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'bool']).columns

        # Converting categorical variables into numerical format
        label_encoders = {}
        categorical_features = ['Patient_Age_Interval', 'Patient_Sex', 'Reporting_Radiologist', 'Cardiomegaly', 'Congestion',
                                'Pleural_Effusion_R', 'Pleural_Effusion_L', 'Atelectasis_R', 'Atelectasis_L']
        for feature in categorical_features:
            le = LabelEncoder()
            X[feature] = le.fit_transform(X[feature])
            label_encoders[feature] = le

        # Creating preprocessing pipelines for both numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combining preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)])

        # Creating the pipeline with preprocessing and model
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', GradientBoostingClassifier())])

        # Splitting the data into training and validation sets
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

        # Training the model
        model.fit(X_train, y_train)

        # Checking the model's performance on the validation set
        validation_score = model.score(X_valid, y_valid)

        # Redefining the categorical transformer with OneHotEncoder
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combining preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)])

        # Updating the model pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', GradientBoostingClassifier())])

        # Training the model
        model.fit(X_train, y_train)

        # Checking the model's performance on the validation set
        validation_score = model.score(X_valid, y_valid)

        # Recreating the preprocessing pipelines for both numerical and categorical data
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))])

        # Combining preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)])

        # Creating the model pipeline
        model = Pipeline(steps=[('preprocessor', preprocessor),
                                ('classifier', GradientBoostingClassifier())])

        # Splitting the data into training and validation sets again
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

        # Training the model
        model.fit(X_train, y_train)

        # Checking the model's performance on the validation set
        validation_score = model.score(X_valid, y_valid)

        # Load the test dataset
        test_file_path = '/mnt/data/full_nolable_task4_test_set_CXR_LabValues_Aachen.xlsx'
        test_df = pd.read_excel(test_file_path)

        # Preprocess the test dataset
        X_test = test_df.drop(['patient_ID', 'Exam_Date', 'Exam_Time', 'split'], axis=1)

        # Converting categorical variables into numerical format
        label_encoders = {}
        categorical_features = ['Patient_Age_Interval', 'Patient_Sex', 'Reporting_Radiologist', 'Cardiomegaly', 'Congestion',
                                'Pleural_Effusion_R', 'Pleural_Effusion_L', 'Atelectasis_R', 'Atelectasis_L']
        for feature in categorical_features:
            le = LabelEncoder()
            X_test[feature] = le.fit_transform(X_test[feature])
            label_encoders[feature] = le

        # Predict probabilities on the test dataset
        test_probabilities = model.predict_proba(X_test)[:,
                             1]  # Probability of class 1 (presence of Pulmonary_Opacities)

        # Creating a dataframe for output
        output_df = pd.DataFrame(
            {'patient_ID': test_df['patient_ID'], 'Pulmonary_Opacities_Probability': test_probabilities})

        # Path for saving the output CSV file
        output_file_path = '/mnt/data/pulmonary_opacities_predictions.csv'
        output_df.to_csv(output_file_path, index=False)



    def main_train_datascientist(self):
        train_file_path = '/PATH/task4_train_set_CXR_LabValues_Aachen.xlsx'
        test_file_path = '/PATH/task4_test_set_CXR_LabValues_Aachen.xlsx'
        train_data = pd.read_excel(train_file_path)
        test_data = pd.read_excel(test_file_path)

        # imputing
        for column in ["Leukocyte", "Procalcitonin", "CRP"]:
            median_val = train_data[column].median()
            train_data[column].fillna(median_val, inplace=True)

        for column in ["Leukocyte", "Procalcitonin", "CRP"]:
            median_val = test_data[column].median()
            test_data[column].fillna(median_val, inplace=True)

        encoders = {}
        categorical_features = ["Patient_Age_Interval", "Patient_Sex", "Reporting_Radiologist"]
        for feature in categorical_features:
            labelencoder = LabelEncoder()
            train_data[feature] = labelencoder.fit_transform(train_data[feature])
            encoders[feature] = labelencoder

        categorical_features = ["Cardiomegaly", "Congestion", "Pleural_Effusion_R", "Pleural_Effusion_L",
                                "Atelectasis_R", "Atelectasis_L"]
        for feature in categorical_features:
            train_data[feature] = train_data[feature].map({'None': 0, '+': 1, '++': 2, '+++': 3, '(+)': 4})
            labelencoder = LabelEncoder()
            train_data[feature] = labelencoder.fit_transform(train_data[feature])
            encoders[feature] = labelencoder


        encoders = {}
        categorical_features = ["Patient_Age_Interval", "Patient_Sex", "Reporting_Radiologist"]
        for feature in categorical_features:
            labelencoder = LabelEncoder()
            test_data[feature] = labelencoder.fit_transform(test_data[feature])
            encoders[feature] = labelencoder

        categorical_features = ["Cardiomegaly", "Congestion", "Pleural_Effusion_R", "Pleural_Effusion_L",
                                "Atelectasis_R", "Atelectasis_L"]
        for feature in categorical_features:
            test_data[feature] = test_data[feature].map({'None': 0, '+': 1, '++': 2, '+++': 3, '(+)': 4})
            labelencoder = LabelEncoder()
            test_data[feature] = labelencoder.fit_transform(test_data[feature])
            encoders[feature] = labelencoder

        numerical_features = ["Cardiomegaly", "Congestion", "Pleural_Effusion_R", "Pleural_Effusion_L", "Atelectasis_R", "Atelectasis_L", "Leukocyte", "Procalcitonin", "CRP"]
        scaler = StandardScaler()
        train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
        test_data[numerical_features] = scaler.fit_transform(test_data[numerical_features])

        # x_train = train_data.drop(columns=["patient_ID", "Exam_Date", "Exam_Time", "Pulmonary_Opacities", "split", "Leukocyte", "Procalcitonin", "CRP"])
        # x_test = test_data.drop(columns=["patient_ID", "Exam_Date", "Exam_Time", "Pulmonary_Opacities", "split", "Leukocyte", "Procalcitonin", "CRP"])
        x_train = train_data.drop(columns=["patient_ID", "Exam_Date", "Exam_Time", "Pulmonary_Opacities", "split"])
        x_test = test_data.drop(columns=["patient_ID", "Exam_Date", "Exam_Time", "Pulmonary_Opacities", "split"])

        # Splitting data into features and target
        y_train = train_data[['Pulmonary_Opacities']]
        y_test = test_data[['Pulmonary_Opacities']]


        print('training started ...\n')

        #########################
        # SVC
        # param_grid = {
        #     'C': [0.1, 1, 10, 100],
        #     'gamma': [1, 0.1, 0.01, 0.001],
        #     'kernel': ['rbf', 'linear']
        # }
        # svc = SVC(probability=True)
        #
        # # Define 10-fold stratified cross-validation
        # stratified_kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        # clf = GridSearchCV(svc, param_grid, cv=stratified_kfold, scoring='accuracy')
        # clf.fit(x_train, y_train)
        # y_pred = clf.predict(x_test)
        # y_pred_proba = clf.predict_proba(y_test)[:, 1]
        #########################

        #########################
        # Adaboost
        # base_clf = DecisionTreeClassifier()
        # ada_clf = AdaBoostClassifier(base_estimator=base_clf)
        #
        # param_grid = {
        #     'base_estimator__max_depth': [1, 2, 3, 4],
        #     'n_estimators': [10, 50, 100, 200],
        #     'learning_rate': [0.001, 0.01, 0.1, 0.5, 1.0]}
        #
        # # Use GridSearchCV
        # grid_search = GridSearchCV(ada_clf, param_grid, scoring='roc_auc', cv=10, verbose=1, n_jobs=-1)
        # grid_search.fit(x_train, y_train)
        #
        # # Get the best model
        # best_ada_clf = grid_search.best_estimator_
        # y_pred = best_ada_clf.predict(x_test)
        # y_pred_proba = best_ada_clf.predict_proba(x_test)[:, 1]
        #########################

        #########################
        # GBM
        # Train a Gradient Boosting Classifier
        clf = GradientBoostingClassifier()
        clf.fit(x_train, y_train)
        y_pred_proba = clf.predict_proba(x_test)[:, 1]
        y_pred = (y_pred_proba > 0.5).astype(np.int32)  # Default threshold
        #########################

        #########################
        # lgbm
        # clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, objective='binary', class_weight='balanced', random_state=42)
        # print(clf.get_params())
        # clf.fit(x_train, y_train)
        # y_pred_proba = clf.predict_proba(x_test)[:, 1]
        # y_pred = (y_pred_proba > 0.5).astype(np.int32)  # Default threshold
        #########################

        #####################
        # # Normal Random Forest classifier
        # clf = RandomForestClassifier(n_estimators=1000, random_state=42)
        # clf.fit(x_train, y_train)
        # # Predict using the best model
        # y_pred = clf.predict(x_test)
        # y_pred_proba = clf.predict_proba(x_test)[:, 1]
        #########################

        #########################
        # # grid search Random Forest classifier
        # # Setting hyperparameters for grid search
        # param_grid = {
        #     'n_estimators': [10, 50, 100, 200],
        #     'max_features': ['sqrt', 'log2'],
        #     'max_depth': [None, 5, 10, 15, 20],
        #     'min_samples_split': [2, 3, 5, 7],
        #     'min_samples_leaf': [2, 3, 4, 5],
        #     'bootstrap': [True, False]}
        #
        # clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, verbose=1, n_jobs=-1, scoring='accuracy', cv=5)
        # clf.fit(x_train, y_train)
        #
        # # print("Best hyperparameters found: ", clf.best_params_)
        # print(clf.get_params())
        # # print(clf.best_estimator_)
        #
        # # Predict using the best model
        # y_pred = clf.predict(x_test)
        # y_pred_proba = clf.predict_proba(x_test)[:, 1]
        #########################

        result_df = pd.DataFrame()
        result_df["probability"] = y_pred_proba
        result_df["ground_truth"] = y_test

        result_file_path = "./datascientist_predictions.csv"
        result_df.to_csv(result_file_path, index=False)

        auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)

        print(auc, accuracy, sensitivity, specificity)





if __name__ == '__main__':
    cohort = Validation()

    # cohort.main_train_GPT_ADA()
    cohort.main_train_datascientist()
