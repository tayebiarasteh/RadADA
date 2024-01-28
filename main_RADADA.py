"""
Created on Jan 27, 2024.
main_RADADA.py

@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""

import pdb
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import norm
import lightgbm as lgb



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




class Tasks():
    def __int__(self):
        pass

    def task1(self):
        # Load the dataset
        file_path = '/mnt/data/CXR LabValues Aachen.xlsx'
        data = pd.read_excel(file_path)

        # Display the first few rows of the dataset to understand its structure
        data.head()

        # Extracting year from the 'Exam Date' column
        data['Year'] = pd.to_datetime(data['Exam Date'], errors='coerce').dt.year

        # Counting the number of exams per year
        utilization_per_year = data['Year'].value_counts().sort_index()

        # Plotting the data
        plt.figure(figsize=(12, 6))
        sns.barplot(x=utilization_per_year.index, y=utilization_per_year.values, palette="viridis")
        plt.title('Radiologic Exams Utilization Rates by Year', fontsize=18)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Number of Exams', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y')

        # Save the plot as a high-quality PNG file
        output_file_path = '/mnt/data/radiologic_exams_utilization_rates.png'
        plt.savefig(output_file_path, format='png', dpi=300)


    def task2(self):
        # Load the dataset
        file_path = '/mnt/data/CXR LabValues Aachen.xlsx'
        data = pd.read_excel(file_path)

        # Display the first few rows of the dataset to understand its structure
        data.head()

        # Mapping severity levels to numerical values
        severity_map = {'None': 0, '+': 1, '++': 2, '+++': 3}

        # Applying the mapping and considering the higher severity between the two sides
        data['Pulmonary Opacities Severity'] = data[['Pulmonary Opacities (r)', 'Pulmonary Opacities (l)']].replace(
            severity_map).max(axis=1)

        # Filtering out the questionable labels
        data_filtered = data[data['Pulmonary Opacities Severity'].notna()]

        # Calculating the mean severity for each combination of age interval and sex
        mean_severity = data_filtered.groupby(['Patient Age Interval', 'Patient Sex'])[
            'Pulmonary Opacities Severity'].mean().unstack()

        mean_severity.reset_index(inplace=True)


    # Function to convert image finding values to binary
    def convert_to_binary(self, val):
        if val in ["None", "(+)"]:
            return 0
        elif val in ["+", "++", "+++"]:
            return 1
        else:
            return np.nan

    def task3(self):
        # Load the dataset
        file_path = '/mnt/data/CXR LabValues Aachen.xlsx'
        df = pd.read_excel(file_path)

        # Display the first few rows of the dataframe to understand its structure
        df.head()

        # Apply the binary conversion to relevant columns
        image_finding_cols = ['Cardiomegaly', 'Congestion', 'Pleural Effusion (r)',
                              'Pleural Effusion (l)', 'Pulmonary Opacities (r)',
                              'Pulmonary Opacities (l)', 'Atelectasis (r)', 'Atelectasis (l)']

        for col in image_finding_cols:
            df[col] = df[col].apply(self.convert_to_binary)

        # Combining pulmonary opacities into one entity
        df['Pulmonary Opacities'] = df[['Pulmonary Opacities (r)', 'Pulmonary Opacities (l)']].max(axis=1)

        # Drop the individual pulmonary opacity columns
        df.drop(['Pulmonary Opacities (r)', 'Pulmonary Opacities (l)'], axis=1, inplace=True)

        # Handle missing values
        # For simplicity, filling missing values with the median for numerical columns and mode for categorical columns
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)

        # Encoding categorical variables
        label_encoder = LabelEncoder()
        df['Patient Age Interval'] = label_encoder.fit_transform(df['Patient Age Interval'])
        df['Patient Sex'] = label_encoder.fit_transform(df['Patient Sex'])
        df['Reporting Radiologist'] = label_encoder.fit_transform(df['Reporting Radiologist'])

        # Selecting features for the analysis
        features = df.drop(['Patient ID', 'Exam Date', 'Exam Time', 'Pulmonary Opacities'], axis=1)
        target = df['Pulmonary Opacities']

        # Logistic regression model
        model = sm.Logit(target, sm.add_constant(features)).fit()

        # Summary of the model
        model_summary = model.summary2()

        # Extracting relevant information from the model summary
        results = model_summary.tables[1].reset_index()
        results.columns = ['Variable', 'Coefficient', 'Std. Error', 'z', 'P-value', 'CI Lower', 'CI Upper']

        # Sorting the results based on the absolute value of the coefficient to show the influence
        results['Abs Coefficient'] = results['Coefficient'].abs()
        sorted_results = results.sort_values(by='Abs Coefficient', ascending=False).drop('Abs Coefficient', axis=1)

        sorted_results.reset_index(drop=True)


    def task4(self):
        todo






class Autonomous():
    def __int__(self):
        pass

    def statistics(self):
        # Load the dataset
        file_path = '/mnt/data/CXR LabValues Aachen.xlsx'
        data = pd.read_excel(file_path)

        # Display the first few rows of the dataset for an initial overview
        data.head()

        # Descriptive statistics for the continuous variables (lab values)
        lab_values_stats = data.describe()

        # Distribution of demographics: Patient Age Interval, Patient Sex
        age_distribution = data['Patient Age Interval'].value_counts()
        sex_distribution = data['Patient Sex'].value_counts()

        lab_values_stats, age_distribution, sex_distribution

        # Analysis of the frequency and distribution of various imaging findings and their severity
        imaging_columns = ['Cardiomegaly', 'Congestion', 'Pleural Effusion (r)', 'Pleural Effusion (l)',
                           'Pulmonary Opacities (r)', 'Pulmonary Opacities (l)', 'Atelectasis (r)', 'Atelectasis (l)']

        # Count the frequency of each severity level for each imaging finding
        imaging_findings_distribution = {col: data[col].value_counts() for col in imaging_columns}

        imaging_findings_distribution



    def correlationanalysis(self):
        # For the purpose of correlation analysis, we need to convert the categorical severity levels into numerical values
        # Mapping: 'None': 0, '(+)': 1, '+': 2, '++': 3, '+++': 4
        severity_mapping = {'None': 0, '(+)': 1, '+': 2, '++': 3, '+++': 4}

        # Apply this mapping to the imaging columns
        for col in imaging_columns:
            data[col] = data[col].map(severity_mapping)

        # Now, let's calculate the correlation matrix
        corr_matrix = data.corr()

        # Plotting the correlation matrix using seaborn heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix of Imaging Findings and Lab Values")
        plt.show()



    def predictive_modeling(self):
        # Selecting features - Demographics and lab values
        features = ['Patient Age Interval', 'Patient Sex', 'Leukocyte Count [x 10^9 / l]', 'Procalcitonin [ng/ml]',
                    'C-Reactive Protein [mg/l]']

        # Data Preprocessing
        # Handling missing values - using mean for continuous variables
        imputer = SimpleImputer(strategy='mean')
        data[features[2:]] = imputer.fit_transform(data[features[2:]])

        # Encoding categorical variables
        data = pd.get_dummies(data, columns=['Patient Age Interval', 'Patient Sex'], drop_first=True)

        # Update features list to include the new dummy variables
        features = list(set(data.columns) - set(imaging_columns) - {'Patient ID', 'Exam Date', 'Exam Time',
                                                                    'Reporting Radiologist'})

        # Splitting the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=0.3,
                                                            random_state=42)

        # Normalizing the feature data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model Building
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        # Predicting on the test set
        y_pred = model.predict(X_test)

        # Model Evaluation
        classification_report_data = classification_report(y_test, y_pred, output_dict=True)

        classification_report_data





if __name__ == '__main__':
    cohort = Validation()
    # cohort.main_train_GPT_ADA()
    cohort.main_train_datascientist()

    taskss = Tasks()
    taskss.task1()
    taskss.task2()
    taskss.task3()
    taskss.task4()

    autonomous = Autonomous()
    autonomous.statistics()
    autonomous.correlationanalysis()
    autonomous.predictive_modeling()
