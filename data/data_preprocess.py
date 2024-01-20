"""
Created on Jan 20, 2024.
data_preprocess.py


@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import pandas as pd
import numpy as np

# Function to categorize age into intervals
def categorize_age(age):
    if age < 20:
        return "0-19"
    elif age < 40:
        return "20-39"
    elif age < 50:
        return "40-49"
    elif age < 60:
        return "50-59"
    elif age < 70:
        return "60-69"
    elif age < 80:
        return "70-79"
    else:
        return "80-112"

def main():
    # Load the dataset
    file_path = '/PATH/original_UKA_master_list.csv'  # Replace with your file path
    data = pd.read_csv(file_path)

    # Modify the 'age' column
    data['age'] = data['age'].apply(lambda x: categorize_age(np.round(x)))

    # Convert 'examination_date' to month/year format
    data['examination_date'] = pd.to_datetime(data['examination_date'], dayfirst=True).dt.strftime('%m/%Y')

    # Convert 'study_time' to time format (HH:MM)
    data['study_time'] = pd.to_datetime(data['study_time'], unit='s').dt.strftime('%H:%M')

    # Remove rows where 'examination_date' is in the year 2020
    data = data[~data['examination_date'].str.endswith('/2020')]

    # Modify 'patient_sex' column
    data['patient_sex'] = data['patient_sex'].replace({'M': 'Male', 'F': 'Female'})

    # Remove rows where 'patient_sex' is 'U' or NaN
    data = data.dropna(subset=['patient_sex'])
    data = data[data['patient_sex'] != 'U']

    # Rename columns
    data = data.rename(columns={
        "age": "Patient Age Interval",
        "examination_date": "Exam Date",
        "study_time": "Exam Time",
        "patient_sex": "Patient Sex",
        "cardiomegaly": "Cardiomegaly",
        "congestion": "Congestion",
        "pleural_effusion_right": "Pleural Effusion (r)",
        "pleural_effusion_left": "Pleural Effusion (l)",
        "pneumonic_infiltrates_right": "Pulmonary Opacities (r)",
        "pneumonic_infiltrates_left": "Pulmonary Opacities (l)",
        "atelectasis_right": "Atelectasis (r)",
        "patient_id": "Patient ID",
        "image_id": "Image ID",
        "LEUK": "Leukocyte Count [x 10^9 / l]",
        "PCTKM": "Procalcitonin [ng/ml]",
        "CRP": "C-Reactive Protein [mg/l]",
        "atelectasis_left": "Atelectasis (l)"
    })

    # Create a mapping for physician names to IDs
    unique_physician_names = data['dignosis_physician_Name'].unique()
    physician_id_mapping = {name: f"RE{str(i+1).zfill(3)}" for i, name in enumerate(unique_physician_names)}

    # Add a new column 'Reporting Radiologist' with the generated IDs
    data['Reporting Radiologist'] = data['dignosis_physician_Name'].map(physician_id_mapping)

    # Define the mapping for the label values
    label_mapping = {
        1: 'None',
        2: '+',
        3: '++',
        4: '+++',
        5: '(+)'
    }

    # List of label columns to replace values in
    label_columns = [
        'Cardiomegaly', 'Congestion', 'Pleural Effusion (r)',
        'Pleural Effusion (l)', 'Pulmonary Opacities (r)',
        'Pulmonary Opacities (l)', 'Atelectasis (r)', 'Atelectasis (l)'
    ]

    # Replace the values for the labels according to the mapping in the specified columns only
    for column in label_columns:
        data[column] = data[column].map(label_mapping)

    # Rename 'image_id' values to a new format
    new_image_ids = [f'image_{str(i+1).zfill(6)}' for i in range(len(data))]
    data['image_id'] = new_image_ids

    # Sort the dataset by 'Exam Date', 'patient_id', and then by 'Exam Time'
    data = data.sort_values(by=['Exam Date', 'Patient ID', 'Exam Time'])

    # Save the modified dataset
    modified_file_path = '/PATH/resssss.csv'  # Define your desired file path
    data.to_csv(modified_file_path, index=False)


if __name__ == "__main__":
    main()
    data_df = pd.read_csv('/PATH/resssss.csv')
    modified_file_path = '/PATH/CXR LabValues Aachen.xlsx'  # Define your desired file path
    data_df.to_excel(modified_file_path, index=False)
