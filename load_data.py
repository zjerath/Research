import pandas as pd
import re

def preprocess_notes(notes):
    # Define patterns to remove
    patterns_to_remove = [
        r"Name:\s+___\s+Unit\s+No:\s+___",
        r"Admission\s+Date:\s+___\s+Discharge\s+Date:\s+___",
        r"Date\s+of\s+Birth:\s+___\s+Sex:\s+[FM]"
    ]

    # Compile regex patterns
    compiled_patterns = [re.compile(pattern) for pattern in patterns_to_remove]

    # Remove matching patterns from notes
    for pattern in compiled_patterns:
        notes = re.sub(pattern, '', notes)

    # Remove extra newlines and leading/trailing spaces
    notes = re.sub(r'\n{2,}', '\n', notes).strip()

    return notes

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

    # preprocess notes column
    merged_df['notes'] = merged_df['notes'].apply(preprocess_notes)

    return merged_df

if __name__ == '__main__':
    preprocessed_data = load_data()
    print(preprocessed_data)
