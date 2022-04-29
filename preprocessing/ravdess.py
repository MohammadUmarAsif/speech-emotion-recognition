import os
import librosa


ROOT = './RAVDESS Data'


def get_files() -> list:
    '''
    Gets file names from dataset

    param: None

    return: file names
    '''

    files = []
    
    for actor_no in range(1, 25):
        path = None
        if actor_no <= 9:
            path = f'{ROOT}/Actor_0{actor_no}'
        else:
            path = f'{ROOT}/Actor_{actor_no}'

        for file_name in os.listdir(path):
            files.append(file_name.split('.')[0])

    return files


def extract_information(files: list, int_to_emotion: dict, int_to_stmt: dict) -> tuple:
    '''
    Extracts labels (emotion, gender, intensity) and details (statement, repitition) 

    param: files = file names
    param: int_to_emotion = mapping integer to emotion
    param: int_to_stmt = mapping integer to statement

    return: labels of samples and details of samples
    '''
    
    label_data = {}
    detail_data = {}
    
    for file_name in files:
        label = file_name.split('-')
        
        emotion = int_to_emotion[int(label[2])]
        gender = 'Female' if int(label[6]) % 2 == 0 else 'Male'
        intensity = 'Strong' if int(label[3]) == 2 else 'Normal'
        
        statement = int_to_stmt[int(label[4])]
        repitition = int(label[5])
        
        label_data[file_name] = [emotion, gender, intensity]
        detail_data[file_name] = [statement, repitition]
        
    return (label_data, detail_data)


def validate_labels(label_data: dict, class_labels: list) -> None:
    """
    Validates the extracted labels from RAVDESS

    param: label_data = labels of samples
    param: class_labels = classes

    return: None
    """

    count = 0

    for key, value in label_data.items():
        category = value[0]
        gender = value[1]
        intensity = value[2]
        
        if category not in class_labels:
            print(f'Invalid (category label): {category} | File: {key}')
            count += 1
        if gender not in ['Male', 'Female']:
            print(f'Invalid (gender label): {gender} | File: {key}')
            count += 1
        if intensity not in ['Normal', 'Strong']:
            print(f'Invalid (intensity label): {gender} | File: {key}')
            count += 1
    
    if count == 0:
        print('All labels are valid.')
    elif count > 0:
        print(f'{count} invalid label(s).')


def load_audio_files(files: list) -> dict:
    '''
    Loads audio data from wav files

    param: files = file names

    return: audio of samples
    '''

    audio_data = {}

    for key in files:  
        actor_no = int(key[-2:])
        file_name = key
        path = None
        
        if actor_no <= 9:
            path = f'{ROOT}/Actor_0{actor_no}/{file_name}.wav'
        else:
            path = f'{ROOT}/Actor_{actor_no}/{file_name}.wav'

        x, sample_rate = librosa.load(path)

        audio_data[key] = [sample_rate, x]
    
    return audio_data

