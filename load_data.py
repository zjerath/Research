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

    # join on SUBJECT_ID
    merged_df = patients.join(discharge.set_index('subject_id'), on='subject_id')

    return merged_df

if __name__ == '__main__':
    preprocessed_data = load_data()
    print(preprocessed_data)