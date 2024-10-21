import os
from pathlib import Path
import re
import json
import numpy as np
import pandas as pd

def clean_text_ecg(text):
    # Remove non-alphanumeric characters (keep letters, numbers, and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9.,;!?\'\"]+', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading and trailing spaces
    return text.strip()

def get_clinical_xray(base_path):
    clinical_xray = {}
    for file_path in Path(base_path).rglob('*.txt'):
        p_number = str(file_path).split("/p")[2].split("/")[0]
        s_number = os.path.splitext(str(file_path).split("/s")[2].split("/")[0])[0]

        key = f'{p_number}_{s_number}'

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        # Finding the sections
        findings_index = content.find("FINDINGS:")
        impression_index = content.find("IMPRESSION:")

        if findings_index != -1 or impression_index != -1:
            findings_section = ""
            impression_section = ""

            if findings_index != -1:
                # Extract from "FINDINGS:" to either "IMPRESSION:" or end of the file
                end_of_findings = content.find("IMPRESSION:", findings_index) if impression_index != -1 else len(content)
                findings_section = content[findings_index:end_of_findings].strip()

            if impression_index != -1:
                # Extract from "IMPRESSION:" to the end of the file
                impression_section = content[impression_index:].strip()

            clinical_xray[key] = findings_section + "\n" + impression_section
        else:
            # If neither section is found, store the entire content
            clinical_xray[key] = content
            
        return clinical_xray
        
# Function to remove '\n' and '_' from a string
def clean_text(text):
    text.replace('\n', ' ')
    # text.replace('_', ' ')
    text = re.sub(r'[_\u2013\u2014\u2015\u2212\uFE58\uFE63\uFF0D]', '', text)  # Remove all underscore-like characters
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    return text.strip() 

def remove_sentences_with_time(text):
    # Regex pattern to match time formats
    time_pattern = r'\d{1,2}:\d{2}(?:\s?[ap]\.?m\.?)?|\d{1,2}\s?[ap]\.?m\.?'
    
    # Regex pattern to split text into sentences
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s'

    # Split text into sentences
    sentences = re.split(sentence_pattern, text)

    # Reconstruct text without sentences containing time
    cleaned_text = ' '.join(sentence for sentence in sentences if not re.search(time_pattern, sentence))

    return cleaned_text


cleaned_texts = {doc_id: clean_text(text) for doc_id, text in clinical_xray.items()}
cleaned_texts = {doc_id: remove_sentences_with_time(text) for doc_id, text in cleaned_texts.items()}

with open('./data/notes_xray.json', 'w') as json_file:
    json.dump(cleaned_texts, json_file, indent=4)

df = pd.read_csv('./data/cxr_ecg_merged_labels_60days.csv')
data = df.values
data = [[item[8], item[9], item[38], item[39], item[14:22], item[24:37], item[7]] for item in data]

labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation',
       'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
       'Lung Opacity', 'No Finding', 'Pleural Effusion', 'Pleural Other',
       'Pneumonia', 'Pneumothorax']

substrings_to_remove = ["FINDING:", "IMPRESSION:"]

# Function to generate X-ray note or label-based note if empty
def generate_xray_note(labels_array, labels):
    present_labels = [labels[i] for i, label_present in enumerate(labels_array) if label_present == 1.0]
    uncertain_labels = [labels[i] for i, label_present in enumerate(labels_array) if label_present == -1.0]
    
    parts = []
    
    if present_labels:
        parts.append(", ".join(present_labels) + " is present")
    
    # If there are uncertain labels, add them to the parts list
    if uncertain_labels:
        parts.append("uncertain finding of " + ", ".join(uncertain_labels))
        
    return ". ".join(parts)
    
def process_item(item, include_special_tokens=False):
    # Extract or generate the X-ray note
    xray_note = item[2] if item[2].strip() else generate_xray_note(item[4], labels)
    xray_note = xray_note.replace("FINDINGS:", "").replace("IMPRESSION:", "")
    # Use the ECG note or a placeholder if not available
    ecg_note = item[3] if item[3].strip() else "ECG note not available."
    # Combine the notes with modality indicators
    # if include_special_tokens:
    #     combined_notes = f"<CLS> <XRAY> {xray_note} <SEP> <ECG> {ecg_note}"
    # else:
    #     combined_notes = f"The report from Xray is: {xray_note.strip()} <SEP> The report from ECG is:{ecg_note.strip()}"
    xray_note = 'The report from Xray is: '+xray_note
    ecg_note = 'The report from ECG is: '+ecg_note
    modified_item = list(item)  # Ensure we have a mutable version of the item
    modified_item[2] = xray_note  # Place combined notes in the X-ray note position
    modified_item[3] = ecg_note
    
    return modified_item

combined_notes_list = [process_item(item) for item in data]

np.save('./data/xray_ecg_notes_labels_combined_60days.npy', combined_notes_list)