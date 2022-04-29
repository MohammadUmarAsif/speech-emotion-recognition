import torch
import numpy as np
from helper.common import shuffle
from helper.preprocessing import save_audio_data


def stratified_test_split(label_data: dict, class_labels: list, split_ratio: float) -> dict:
    '''
    Performs a split of data into train and test lists

    param: label_data = labels of samples
    param: class_labels = classes
    param: split_ratio = % of split

    return: training samples and test samples
    '''

    shuffled_data = shuffle(list(label_data.keys()))
    ordered_data = {}
    for label in class_labels:
        ordered_data[label] = []
    
    # Arranging the files according to emotion label
    for key in shuffled_data:
        value = label_data[key]
        ordered_data[value[0]].append(key)

    split_data = {'train': [], 'test': []}
    
    # Equal split of files from each emotion
    for key, value in ordered_data.items():
        size = len(value)        
        indices = torch.randperm(size).tolist()
        split = int(np.floor(split_ratio * size))
        train_idx, test_idx = indices[split:], indices[:split]

        for idx in train_idx: 
            file_name = value[idx]
            split_data['train'].append(file_name)
        
        for idx in test_idx: 
            file_name = value[idx]
            split_data['test'].append(file_name)
        
    return split_data


def stratified_valid_split(train_subset: list, whole_data: dict, label_data: dict, class_labels: list, split_ratio: float) -> list:
    '''
    Performs a split of data into train and validation lists

    param: train_subset = training samples which are split
    param: whole_data = data in terms of which amount of split calculated
    param: label_data = labels of samples
    param: class_labels = classes
    param: split_ratio = % of split

    return: validation samples
    '''

    if type(whole_data) == dict:
        whole_data = whole_data.keys()

    shuffled_data = shuffle(list(whole_data))
    ordered_data = {}
    label_amount = {}

    for label in class_labels:
        ordered_data[label] = []
        label_amount[label] = 0
    
    # Arranging the files according to emotion label
    for key in shuffled_data:
        value = label_data[key]
        if key in train_subset:
            ordered_data[value[0]].append(key)
        label_amount[value[0]] = label_amount.get(value[0], 0) + 1
    
    valid_data = []
    
    # Equal split of files from each emotion
    for key, value in ordered_data.items():
        size = label_amount[key]   
        indices = torch.randperm(len(value)).tolist()
        split = int(np.floor(split_ratio * size))
        valid_idx = indices[:split]

        for idx in valid_idx: 
            session_name = value[idx]
            valid_data.append(session_name)
           
    # Remove common files
    for key in valid_data:
        train_subset.remove(key)
    
    return valid_data


def make_k_folds(train_data: list, label_data: dict, class_labels: list, k: int) -> dict:
    '''
    Makes k number of folds from the train data

    param: train_data = training samples
    param: label_data = labels of samples
    param: class_labels = classes
    param: k = number of folds

    return k folds of samples
    '''
    
    train_data = list(train_data)
    train_subset = list(train_data)
    cross_data = {}
    split_ratio = 1/k

    # k-1 folds
    for i in range(k-1):
        cross_data[str(i+1)] = stratified_valid_split(train_subset, train_data, label_data, class_labels, split_ratio)

    # kth fold
    cross_data[str(k)] = train_subset

    return cross_data


def get_folds(cross_data: dict, fold_num: int) -> tuple:
    '''
    Creates the train set and valid set by combining folds

    param: cross_data = k folds of samples
    param: fold_num = key of validation fold

    return: training samples and validation samples
    '''
    
    valid_fold = cross_data[str(fold_num)]
    train_folds = []

    for i in range(len(cross_data)):
        if i+1 != fold_num:
            train_folds.extend(cross_data[str(i+1)])

    return (train_folds, valid_fold)


def create_subsets(root: str, audio_data: dict, split_type: str, size: int) -> int:
    '''
    Creates subsets of audio data and saves them

    param: root = root folder name
    param: audio_data = audio of samples
    param: split_type = type of split
    param: size = maximum size of subset

    return: number of subsets
    '''
  
    num_subsets = int(np.ceil(len(audio_data)/size))

    for i in range(0, num_subsets):
        audio_subset = list(audio_data.items())[i*size:(i+1)*size]
        save_audio_data(root, dict(audio_subset), split_type, f'temp{i}')

    return num_subsets

    