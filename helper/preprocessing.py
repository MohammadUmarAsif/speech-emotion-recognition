import os
import librosa
import numpy as np
import pandas as pd


def get_numerical_stats(df: pd.DataFrame, columns: list) -> dict:
    '''
    Calculates min, max, diff, mean, and std on dataframe columns
    
    param: df = dataframe
    param: columns = columns of dataframe
    
    return: all stats
    '''
    
    stats = {}
    stat_min = []
    stat_max = []
    stat_mean = []
    stat_std = []
    
    for col in columns:
        stat_min.append(df[col].min())
        stat_max.append(df[col].max())
        stat_mean.append(df[col].mean())
        stat_std.append(df[col].std())
    
    stat_diff = [stat_max[i] - stat_min[i] for i in range(len(stat_min))]

    stats['min'] = stat_min
    stats['max'] = stat_max
    stats['diff'] = stat_diff
    stats['mean'] = stat_mean
    stats['std'] = stat_std
    
    return stats


def numerical_scaler(label_data: dict, stats: dict) -> dict:
    '''
    Applies a scaling to numerical values

    param: label_data = labels of samples
    param: stats = min and max values
    
    return: scaled values
    '''
    
    scaled_dict = {}

    for key in label_data.keys():
        scaled_dict[key] = []

    for key, value in label_data.items():
        scaled_dict[key].append((value[2] - stats['min'][0]) / stats['diff'][0])
        scaled_dict[key].append((value[2] - stats['mean'][0]) / stats['std'][0])

        scaled_dict[key].append((value[3] - stats['min'][1]) / stats['diff'][1])
        scaled_dict[key].append((value[3] - stats['mean'][1]) / stats['std'][1])

        scaled_dict[key].append((value[4] - stats['min'][2]) / stats['diff'][2])
        scaled_dict[key].append((value[4] - stats['mean'][2]) / stats['std'][2])

    return scaled_dict


def custom_round(x: float, prec: int, base: float) -> float:
    '''
    Rounds the input

    param: x = number to be rounded 
    param: prec = number of decimal places
    param: base = base for rounding

    return: rounded number
    '''

    return round(base * round(float(x)/base), prec)


def class_to_int(class_labels: list, name: str) -> dict:
    '''
    Maps the labels to integers

    param: class_labels = classes
    param: name = name of dataset

    return: mapping of labels to integers
    '''

    class_data = {label: idx for idx, label in enumerate(class_labels)}
    class_data['Male'] = 0
    class_data['Female'] = 1
    
    if name == 'ravdess':
        class_data['Normal'] = 0
        class_data['Strong'] = 1

    return class_data


def create_dir(root: str, class_labels: list, name: str) -> None:
    '''
    Creates directories for dataset

    param: root = root folder name
    param: class_labels = classes
    param: name = name of dataset

    return: None
    '''
    
    if not os.path.isdir(root):
        os.mkdir(root)
        
        if name == 'iemocap':
            os.mkdir(f'{root}/Train')
            os.mkdir(f'{root}/Test')
            os.mkdir(f'{root}/Audio')
            os.mkdir(f'{root}/Audio/Train')
            os.mkdir(f'{root}/Audio/Test')

            for label in class_labels:
                os.mkdir(f'{root}/Train/{label}')
                os.mkdir(f'{root}/Test/{label}')

        elif name == 'ravdess':
            os.mkdir(f'{root}/Test')
            os.mkdir(f'{root}/Audio')
            os.mkdir(f'{root}/Audio/Test')
            for label in class_labels:
                os.mkdir(f'{root}/Test/{label}')        


def trim_silence(x: list, top_db: int) -> list:
    '''
    Trims the silence from input signal

    param: x = input signal
    param: top_db = threshold in db below which is considered silence

    return: trimmed signal
    '''
    
    trimmed_signal = librosa.effects.trim(x, top_db = top_db, ref = np.max)
    return trimmed_signal[0]


def drop_labels(class_labels: list, to_drop: list, label_data: dict, detail_data: dict, files: list) -> None:
    '''
    Removes information related to labels that are dropped/not required

    param: class_labels = classes
    param: to_drop = labels to drop
    param: label_data = labels of samples
    param: detail_data = details of samples
    param: files = file names

    return: None
    '''
    
    for label in to_drop:
        class_labels.remove(label)

    dropped_keys = []
    for key, value in label_data.items():
        if value[0] in to_drop:
            dropped_keys.append(key)

    for key in dropped_keys:
        label_data.pop(key)
        detail_data.pop(key)
        files.remove(key)


def compute_audio_length(audio_data: dict, detail_data: dict) -> None:
    '''
    Calculates the audio length

    param: audio_data = audio of samples
    param: detail_data = details of samples

    return: None
    '''

    for key, value in audio_data.items():
        sample_rate, x = value[0], value[1]

        duration = len(x)/sample_rate

        detail_data[key].insert(0, custom_round(duration, 4, 0.0001))


def split_audio(x: list, sample_rate: int, max_duration: float) -> list:
    '''
    Splits audio into samples based on max duration

    param: x = input signal
    param: sample_rate = sample rate of signal
    param: max_duration = maximum duration of each sample

    return: audio samples
    '''

    size = max_duration*sample_rate
    audio_limits = []
    
    # Get limits of each sample
    for i in range(0, int(np.ceil(len(x)/size))):
        limit = [i*size, (i+1)*size]
        
        if len(x) < limit[1]:
            limit[1] = len(x)

        audio_limits.append(limit)

    samples = []

    # Extract the samples
    for limit in audio_limits:
        samples.append(x[limit[0]:limit[1]])
    
    return samples


def divide_audio_data(audio_data: dict, split_data: dict) -> tuple:
    '''
    Divide the audio samples into train and test sets

    param: audio_data = audio of samples
    param: split_data = split samples

    return: train audio samples and test audio samples
    '''
    
    train_audio_data = {}
    test_audio_data = {}

    for key in split_data['train']:
        train_audio_data[key] = audio_data[key]

    for key in split_data['test']:
        test_audio_data[key] = audio_data[key]
    
    return (train_audio_data, test_audio_data)


def trim_audio_files(audio_data: dict, top_db: int) -> dict:
    '''
    Trims audio data based on threshold

    param: audio_data = audio of samples
    param: top_db = threshold in db below which is considered silence

    return: trimmed audio of samples
    '''
    trimmed_audio = {}

    for key, value in audio_data.items():  
        x = trim_silence(value[1], top_db)
        trimmed_audio[key] = [value[0], x]

    return trimmed_audio


def save_audio_data(root: str, audio_data: dict, split_type: str, name: str) -> None:
    '''
    Saves audio data dictionary in npy format

    param: root = root folder name
    param: audio_data = audio of samples
    param: split_type = type of split
    param: name = name of subset

    return: None
    '''
    
    path = f'{root}/Audio/{split_type.capitalize()}/{name}.npy'
    np.save(path, audio_data)

