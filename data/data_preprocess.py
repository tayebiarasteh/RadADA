"""
Created on Jan 20, 2024.
data_preprocess.py


@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import pandas as pd
import numpy as np
from scipy.stats import mode
from tqdm import tqdm

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


# Function to perform majority vote
def majority_vote(values):
    # Count the occurrences of each value
    counts = values.value_counts()
    # Find the most frequent value
    max_count = counts.max()
    # Filter values that have the max count
    most_frequent = counts[counts == max_count]
    # Return the first one in case of tie
    return most_frequent.index[0]


def main():
    # Load the dataset
    file_path = '/PATH.csv'  # Replace with your file path
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
    df = data.sort_values(by=['Exam Date', 'Patient ID', 'Exam Time'])

    # Columns for majority vote
    vote_cols = ['Cardiomegaly', 'Congestion', 'Pleural Effusion (r)', 'Pleural Effusion (l)',
                 'Pulmonary Opacities (r)', 'Pulmonary Opacities (l)', 'Atelectasis (r)', 'Atelectasis (l)']

    # Columns for mean calculation
    mean_cols = ['Leukocyte Count [x 10^9 / l]', 'Procalcitonin [ng/ml]', 'C-Reactive Protein [mg/l]']

    # Processing the data
    grouped = df.groupby(df['Patient ID'])

    # Majority vote for specified columns
    for col in tqdm(vote_cols):
        df[col] = grouped[col].transform(lambda x: majority_vote(x))

    # Mean for lab values columns
    for col in mean_cols:
        df[col] = grouped[col].transform(lambda x: x.mean() if not x.isna().all() else None)

    # Keeping only one row per patient
    df = df.drop_duplicates(subset=['Patient ID'])

    # Saving to Excel
    output_file = '/PATH.xlsx'  # Define your desired file path
    df.to_excel(output_file, index=False)

    print(f"Processed dataset saved to {output_file}")



def sort_dataset(file_path):
    # Read the dataset
    df = pd.read_excel(file_path)

    # Convert 'Exam Date' to datetime format (assuming the format is MM/YYYY)
    df['Exam Date'] = pd.to_datetime(df['Exam Date'], format='%m/%Y')

    # Sort the dataframe by 'Exam Date'
    sorted_df = df.sort_values(by='Exam Date')

    # Revert 'Exam Date' back to month/year format
    sorted_df['Exam Date'] = sorted_df['Exam Date'].dt.strftime('%m/%Y')

    # Save the sorted dataframe to a new Excel file
    sorted_output_file = '/PATH.xlsx'
    sorted_df.to_excel(sorted_output_file, index=False)

    print(f"Sorted dataset saved to {sorted_output_file}")


if __name__ == "__main__":
    # main()
    # data_df = pd.read_csv('/PATH/resssss.csv')

    new_dataset_path = '/PATH.xlsx'  # Define your desired file path
    sort_dataset(new_dataset_path)
