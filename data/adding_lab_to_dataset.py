"""
Created on Jan 20, 2024.
adding_lab_to_dataset.py


@author: Soroosh Tayebi Arasteh <soroosh.arasteh@rwth-aachen.de>
https://github.com/tayebiarasteh/
"""
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
import pdb

# Load the datasets

original_uka_df = pd.read_csv('/original_UKA_master_list.csv')
radiologie_truhn_df = pd.read_csv('/Radiologie_Truhn_Laborwerte.csv')
updated_file_path = '/finallll.csv'


# Function to combine date and time into a single datetime object
def combine_date_time(row):
    try:
        if isinstance(row['study_time'], str):
            study_time = datetime.strptime(row['study_time'], '%H%M%S.%f').time()
        else:
            study_time_str = str(int(row['study_time']))
            study_time = datetime.strptime(study_time_str, '%H%M%S').time()
    except:
        return pd.NaT

    try:
        exam_date = datetime.strptime(row['examination_date'], '%d/%m/%Y')
        return datetime.combine(exam_date, study_time)
    except:
        return pd.NaT


# Apply the function to the original UKA dataset
original_uka_df['exam_datetime'] = original_uka_df.apply(combine_date_time, axis=1)

# Convert 'Date' in Radiologie Truhn Laborwerte dataset to datetime
radiologie_truhn_df['Date'] = pd.to_datetime(radiologie_truhn_df['Date'], format='%Y.%m.%d %H:%M:%S')

# Handle non-numeric lab values in 'Word_txt'
radiologie_truhn_df['Word_txt'] = pd.to_numeric(radiologie_truhn_df['Word_txt'], errors='coerce')


# Function to find the closest lab value for each parameter
def find_closest_lab_values(row, lab_df, parameters):
    reception_id = row['reception_id']
    exam_time = row['exam_datetime']
    window_start = exam_time - timedelta(hours=12)
    window_end = exam_time + timedelta(hours=12)

    # Filter lab values for the same reception_id within the time window
    lab_values = lab_df[(lab_df['reception_id'] == reception_id) &
                        (lab_df['Date'] >= window_start) &
                        (lab_df['Date'] <= window_end)]

    # Find the closest value for each parameter
    closest_values = {}
    for param in parameters:
        param_values = lab_values[lab_values['Analytic'] == param]
        if not param_values.empty:
            param_values['time_diff'] = param_values['Date'].apply(lambda x: abs((x - exam_time).total_seconds()))
            closest_values[param] = param_values.sort_values('time_diff').iloc[0]['Word_txt']
        else:
            closest_values[param] = None

    return pd.Series(closest_values)


# Define the parameters to look for
parameters = ['LEUK', 'PCTKM', 'CRP']

# Apply the function to find the closest lab values with tqdm progress bar
tqdm.pandas(desc="Processing rows")
closest_values_df = original_uka_df.progress_apply(find_closest_lab_values, axis=1,
                                                   args=(radiologie_truhn_df, parameters))

# Update the original dataset with the found values
for param in parameters:
    original_uka_df[param] = closest_values_df[param]

# Save the updated dataset
original_uka_df.to_csv(updated_file_path, index=False)
