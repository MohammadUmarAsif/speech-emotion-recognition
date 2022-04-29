import os
import librosa
import numpy as np
from helper.preprocessing import custom_round, split_audio


ROOT = './IEMOCAP Data'


def extract_information() -> tuple:
    '''
    Extracts details from file (audio duration and transcription)
    
    param: None

    return: details of samples and file names
    '''

    details_data = {}
    files = []

    for session_no in range(1, 6):
        path = f'{ROOT}/Session{session_no}/dialog/transcriptions'

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)

            with open(file_path, 'r') as file:  
                for line in file:
                    if (line[0] == 'S'):
                        tokens = line.split(' ')
                        session_name = tokens[0]

                        # Skipping utterances that are not present in .wav format
                        if 'X' in session_name:
                            continue
                        
                        transcription = ' '.join(tokens[2:]).strip('\n')

                        details_data[session_name] = [transcription]
                        files.append(session_name)
    
    return (details_data, files)


def extract_labels() -> dict:
    '''
    Extracts labels from file (categorical, gender and dimensional)

    param: None

    return: labels of samples
    '''

    labels_data = {}
    
    for session_no in range(1, 6):
        path = f'{ROOT}/Session{session_no}/dialog/EmoEvaluation'

        for file_name in os.listdir(path):
            file_path = os.path.join(path, file_name)

            with open(file_path, 'r') as file:
                extracting = False
                attr_eval = None
                attr_eval_count = None
                cat_eval_count = {}
                session_name = None

                for line in file:
                    
                    # Encountered a block of text to extract
                    if (line[0] == '['):
                        extracting = True
                        tokens = line.split('\t')
                        session_name = tokens[1]
                        attr_eval = np.zeros(3)
                        attr_eval_count = 0
                        continue

                    if (extracting):

                        # Categorical label
                        if (line[0] == 'C'):
                            tokens = line.split('\t')
                            label_list = tokens[1].strip(';')

                            # Storing frequency of each label
                            for label in label_list.split(';'):
                                cat_eval_count[label.strip()] = cat_eval_count.get(label.strip(), 0) + 1

                        # Dimensional label
                        elif (line[0] == 'A'):
                            tokens = line.split('\t')
                            attr_list = tokens[1].strip(';').split(';')

                            # Storing val, act, dom labels
                            for idx, attr in enumerate(attr_list):
                                attr = attr.strip()

                                if attr[-1].isnumeric():
                                    attr_eval[idx] += int(attr[-1])
                                else:
                                    attr_eval[idx] += 1

                            attr_eval_count += 1

                        # End of extraction block
                        elif (line[0] == '\n'):   
                            category = max(cat_eval_count, key=cat_eval_count.get)

                            attr_avg = attr_eval/attr_eval_count
                            for idx, attr in enumerate(attr_avg):
                                attr_avg[idx] = custom_round(attr, 1, 0.1)

                            gender = 'Female' if session_name[-4] == 'F' else 'Male'
                            
                            labels_list = [category, gender]
                            for attr in attr_avg:
                                labels_list.append(attr)
                            
                            labels_data[session_name] = labels_list
                            
                            # Reset variables
                            extracting = False
                            cat_eval_count.clear()
                            attr_eval = None
                            attr_eval_count = None
                            session_name = None
    return labels_data


def validate_labels(label_data: dict, class_labels: list) -> None:
    """
    Validates the extracted labels from IEMOCAP

    param: label_data = labels of samples
    param: class_labels = classes

    return: None
    """

    count = 0

    for key, value in label_data.items():
        category = value[0]
        gender = value[1]
        val = value[2]
        act = value[3]
        dom = value[4]
        
        if category not in class_labels:
            print(f'Invalid (category label): {category} | File: {key}')
            count += 1
        if gender not in ['Male', 'Female']:
            print(f'Invalid (gender label): {gender} | File: {key}')
            count += 1
        if not (val >= 1 and val <= 6):
            print(f'Invalid (valence value): {val} | File: {key}')
            count += 1
        if not (act >= 1 and act <= 6):
            print(f'Invalid (activation value): {act} | File: {key}')
            count += 1
        if not (dom >= 1 and dom <= 6):
            print(f'Invalid (dominance value): {dom} | File: {key}')
            count += 1
    
    if count == 0:
        print('All labels and values are valid.')
    elif count > 0:
        print(f'{count} invalid label(s) and/or value(s).')
   
        
def load_audio_files(files: list) -> dict:
    '''
    Loads audio data from wav files

    param: files = file names

    return: audio of samples
    '''

    audio_data = {}

    for key in files:        
        session_no = key[4]
        folder_name = key[:-5]
        file_name = key

        path = f'{ROOT}/Session{session_no}/sentences/wav/{folder_name}/{file_name}.wav'

        x, sample_rate = librosa.load(path)

        audio_data[key] = [sample_rate, x]

    return audio_data


def split_audio_files(audio_data: dict, label_data: dict, detail_data: dict, scaled_vad: dict, max_duration: float) -> tuple:
    '''
    Splits the audio files

    param: audio_data = audio of samples
    param: label_data = labels of samples
    param: detail_data = details of samples
    param: scaled_vad = scaled numerical values
    param: max_duration = maximum duration of each sample

    return: audio of samples, labels of samples, and details of samples
    '''

    new_audio_data = {}
    new_label_data = {}
    new_detail_data = {}
    new_scaled_vad = {}

    for key, value in audio_data.items():        
        sample_rate, x = value[0], value[1]

        samples = split_audio(x, sample_rate, max_duration)
        
        for idx, sample in enumerate(samples):
            new_key = key + '_split' + str(idx+1)
        
            new_audio_data[new_key] = [sample_rate, sample]
            new_label_data[new_key] = list(label_data[key])
            new_detail_data[new_key] = list(detail_data[key])
            new_scaled_vad[new_key] = list(scaled_vad[key])
    
    return (new_audio_data, new_label_data, new_detail_data, new_scaled_vad)


def drop_audio_samples(audio_data: dict, label_data: dict, detail_data: dict, scaled_vad: dict, min_duration: float) -> None:
    '''
    Removes the audio samples which are splits and have duration lesser than threshold
    
    param: audio_data = audio of samples
    param: label_data = labels of samples
    param: detail_data = details of samples
    param: scaled_vad = scaled numerical values
    param: min_duration = threshold in seconds below which sample is dropped

    return: None
    '''
    
    to_drop = []
    
    for key, value in detail_data.items():
        if value[0] < min_duration and int(key[-1]) != 1:
            to_drop.append(key)
    
    for key in to_drop:
        audio_data.pop(key)
        label_data.pop(key)
        detail_data.pop(key)
        scaled_vad.pop(key)

