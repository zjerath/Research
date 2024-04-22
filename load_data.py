import pandas as pd

def load_data():
    # load csv file to pandas df
    patients = pd.read_csv('files/patients.1000.csv')
    discharge = pd.read_csv('files/discharge.1000.csv')

    # filter columns
    patient_columns = ['subject_id', 'anchor_age']
    patients = patients[patient_columns]
    discharge_columns = ['subject_id', 'notes']
    discharge = discharge[discharge_columns]

    # convert notes to string
    discharge['notes'] = discharge['notes'].astype(str)

    # filter patient to only include subject IDs present in discharge
    patients = patients[patients['subject_id'].isin(discharge['subject_id'])]

    # join on SUBJECT_ID
    merged_df = patients.merge(discharge, on='subject_id', how='inner')

    return merged_df

if __name__ == '__main__':
    preprocessed_data = load_data()
    print(preprocessed_data)
