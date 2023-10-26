import pandas as pd

def load_data():
    # load csv file to pandas df
    filename_notes = './files/NOTEEVENTS.small.csv'
    df_notes = pd.read_csv(filename_notes, engine='python', on_bad_lines='skip')

    filename_patients = './files/PATIENTS.csv'
    df_patients = pd.read_csv(filename_patients, engine='python', on_bad_lines='skip')

    # filter columns
    selected_columns_notes = ['SUBJECT_ID', 'CHARTDATE', 'TEXT', 'CATEGORY']
    selected_df_notes = df_notes[selected_columns_notes]

    selected_columns_patients = ['SUBJECT_ID', 'DOB']
    selected_df_patients = df_patients[selected_columns_patients]

    # filter category
    discharge_summaries = selected_df_notes[selected_df_notes['CATEGORY'] == 'Discharge summary']

    # join on SUBJECT_ID
    merged_df = discharge_summaries.merge(selected_df_patients, on='SUBJECT_ID', how='inner')

    # convert DOB and CHARTDATE to datetime
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'])
    merged_df['CHARTDATE'] = pd.to_datetime(merged_df['CHARTDATE'])

    # calculate age
    def calculate_age(row):
        age = row['CHARTDATE'].year - row['DOB'].year - ((row['CHARTDATE'].month, row['CHARTDATE'].day) < (row['DOB'].month, row['DOB'].day))
        return age

    merged_df['AGE'] = merged_df.apply(calculate_age, axis=1)

    # simplified
    simple_selected_columns = ['SUBJECT_ID', 'TEXT', 'AGE']
    simple_merged_df = merged_df[simple_selected_columns]

    return simple_merged_df

if __name__ == '__main__':
    preprocessed_data = load_data()
    print(preprocessed_data)