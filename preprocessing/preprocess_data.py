import numpy as np
import pandas as pd
import glob
import os
from datetime import datetime
import json

#cxr preprocess
cxr_metadata = pd.read_csv('./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-metadata.csv')

cxr_metadata['StudyDate'] = pd.to_datetime(cxr_metadata['StudyDate'], format='%Y%m%d').dt.date
cxr_metadata = cxr_metadata[(cxr_metadata['ViewPosition'] == 'PA') | (cxr_metadata['ViewPosition'] == 'AP')]
cxr_metadata = cxr_metadata.reset_index(drop=True)

cxr_splits_csv = pd.read_csv('./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-split.csv')
cxr_metadata_split = cxr_metadata.merge(cxr_splits_csv[['dicom_id', 'split']], on='dicom_id', how='inner')
cxr_chexpert_csv = pd.read_csv('./mimic-cxr-jpg-2.0.0.physionet.org/mimic-cxr-2.0.0-chexpert.csv')
cxr_chexpert_csv = cxr_chexpert_csv.fillna(0)
cxr_metadata_chexpert = cxr_metadata_split.merge(cxr_chexpert_csv, on = ['subject_id', 'study_id'], how = 'inner')

cxr_files= []
# Open the file in read mode
with open('../data/cxr_paths.txt', 'r') as file: #file with mimic cxr paths
    # Read each line in the file and append it to the list
    for line in file:
        # Strip newline characters from the end of each line
        cxr_files.append(line.strip())
        
basename_path_dict = dict()
for path in cxr_files:
    basename_path_dict[os.path.splitext(os.path.basename(path))[0]] = path

cxr_metadata_chexpert['path'] = cxr_metadata_chexpert['dicom_id'].map(basename_path_dict)

with open('../data/notes_xray_path.json', 'r') as file: #file with mimic xray path and notes
    xray_dict = json.load(file)

cxr_metadata_chexpert['XrayReport'] = cxr_metadata_chexpert['path'].map(xray_dict)
cxr_metadata_chexpert['XrayReport'].fillna('', inplace=True)


###################### ecg preprocess ###########################
ecg_metadata = pd.read_csv('./physionet.org/files/mimic-iv-ecg/1.0/machine_measurements.csv')
ecg_metadata = ecg_metadata.fillna('')
ecg_metadata['merged_report'] = ecg_metadata[['report_0', 'report_1', 'report_2', 'report_3', 'report_4', 'report_5', 'report_6']].apply(lambda x: ' '.join(x), axis=1)
# Drop the original 'report_0' to 'report_6' columns
ecg_metadata = ecg_metadata.drop(ecg_metadata.columns[4:22], axis = 1)

ecg_records = pd.read_csv('./physionet.org/files/mimic-iv-ecg/1.0/record_list.csv')
# Convert the 'Datetime' column to datetime objects
ecg_records['ecg_time'] = pd.to_datetime(ecg_records['ecg_time'])

# Extract the date part
ecg_records['ecg_time'] = ecg_records['ecg_time'].dt.date
ecg_records['subject_id'] = ecg_records['subject_id'].astype(int)
ecg_metadata['subject_id'] = ecg_metadata['subject_id'].astype(int)

ecg_merged_df = ecg_metadata.merge(ecg_records[['subject_id', 'study_id','path']], on = ['subject_id', 'study_id'], how = 'inner')
ecg_merged_df['ecg_time'] = pd.to_datetime(ecg_merged_df['ecg_time'])


############## merge ecg and cxr ################

cxr_ecg_merged = pd.merge(cxr_metadata_chexpert, ecg_merged_df, on = 'subject_id')
cols = ['dicom_id', 'subject_id', 'study_id_x', 'study_id_y',
       'StudyDate', 'StudyTime', 'ecg_time','split', 'Atelectasis',
       'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
       'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding',
       'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax',
       'Support Devices', 'path_x','path_y',
       'merged_report', 'XrayReport']

cxr_ecg_merged = cxr_ecg_merged[cols]
cxr_ecg_merged['StudyDate'] = pd.to_datetime(cxr_ecg_merged['StudyDate'])
cxr_ecg_merged['time_diff'] = (cxr_ecg_merged['StudyDate'] - cxr_ecg_merged['ecg_time']).dt.days
cxr_ecg_merged['time_diff'] =cxr_ecg_merged['time_diff'].fillna(999)
filtered_df = cxr_ecg_merged[abs(cxr_ecg_merged['time_diff']) <=60]

error_df = pd.read_csv('./data/cxr_ecg_merged_labels_60days_error.csv')
error_ids = np.unique(error_df['study_id_y'])
filtered_df = filtered_df[~filtered_df['study_id_y'].isin(error_ids)]
filtered_df.to_csv('./data/cxr_ecg_merged_labels_60days.csv')