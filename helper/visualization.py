import copy
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from network.batch import apply_transforms
from PIL import Image
from IPython.display import display


def get_dataframe(data: dict, name: str, info_type: str) -> pd.DataFrame:
    '''
    Converts a dictionary to a pandas dataframe

    param: data = data to convert
    param: name = name of dataset
    param: info_type = type of information the data holds

    return: dataframe of input data
    '''
    
    columns = None

    if name == 'iemocap':
        if info_type == 'label':
                columns = ['File Name', 'Emotion', 'Gender', 'Valence', 'Activation', 'Dominance']
        elif info_type == 'detail':
            if len(list(data.values())[0]) == 3:
                columns = ['File Name', 'Trimmed Duration', 'Original Duration', 'Transcription']
            else:
                columns = ['File Name',  'Transcription']
        elif info_type == 'scaled':
                columns = ['File Name', 'Valence - MM', 'Valence - Std', 'Activation - MM', 'Activation - Std', 'Dominance - MM', 'Dominance - Std']
    
    elif name == 'ravdess':
        if info_type == 'label':
            columns = ['File Name', 'Emotion', 'Gender', 'Intensity']
        elif info_type == 'detail':
            if len(list(data.values())[0]) == 4:
                columns = ['File Name', 'Trimmed Duration', 'Original Duration', 'Statement', 'Repitition']
            else:
                columns = ['File Name', 'Statement', 'Repitition']
    
    data_dic = {}
    for col in columns:
        data_dic[col] = []

    for key, value in data.items():
        for i in range(len(columns)):
            if i == 0:
                data_dic[columns[i]].append(key)
            else:
                data_dic[columns[i]].append(value[i-1])

    return pd.DataFrame(data_dic, columns=columns)


def show_dataframe(df: pd.DataFrame, info_type: str, num_rows: int) -> None:
    '''
    Displays rows of a dataframe

    param: df = dataframe
    param: info_type = type of information the dataframe holds
    param: num_rows = number of rows to display

    return: None
    '''

    print(f'\n-------------{info_type.capitalize()} Data-------------')
    display(df[:num_rows])


def hide_empty(plot_idx: int, num_rows: int, num_cols: int) -> None:
    '''
    Hide remaining empty subplots

    param: plot_idx = index of subplot
    param: num_rows = number of rows
    param: num_cols = number of columns

    return: None
    '''
    
    while plot_idx <= num_rows*num_cols:
        ax = plt.subplot(num_rows, num_cols, plot_idx)
        ax.set_visible(False)
        plot_idx += 1


def get_feature_maps(root: str, model, split_data: dict, split_type: str, label_data: dict, transforms: tuple) -> tuple:
    '''
    Returns the transformed image, attention weights, and feature maps for a random image

    param: root = root folder name
    param: model = trained model
    param: split_data = split samples
    param: split_type = type of split
    param: label_data = labels of samples
    param: transforms = transforms

    return: required data
    '''
    
    file_name = np.random.choice(split_data[split_type])
    emotion = label_data[file_name][0]
    path = f'{root}/{split_type.capitalize()}/{emotion}/{file_name}/{file_name}_aug1.png'
    img = Image.open(path).convert(transforms[3])
    transformed_img = apply_transforms(img, transforms[0], transforms[1], transforms[2], transforms[3])
    transformed_img = transformed_img.view(1, -1, transforms[0][0], transforms[0][1])
    
    _, attention, feature_maps = model.forward(transformed_img)

    return (img, transformed_img, attention, feature_maps)


def combine_data(meta_root: str, checkpoint: dict, min: int, max: int) -> dict:
    '''
    Combines the data (plots and metrics) from all sessions

    param: meta_root = root folder name
    param: checkpoint = data from first session
    param: min = checkpoint number from which to start
    param: max = checkpoint number at which to end
    
    return: combined data
    '''

    all_checkpoints = copy.deepcopy(checkpoint)
    for i in range(min + 1, max + 1):
        checkpoint = torch.load(f'{meta_root}/Checkpoint-{i}.pt', map_location = torch.device('cpu'))
        for data_idx, data_list in enumerate(checkpoint['plotting data']):
            for epoch_list in data_list:
                if data_idx == 2:
                    all_checkpoints['plotting data'][data_idx].append(epoch_list)
                else:
                    all_checkpoints['plotting data'][data_idx].append([[]]*len(checkpoint['plotting data'][data_idx][0]))
                for idx, data in enumerate(epoch_list):
                    all_checkpoints['plotting data'][data_idx][-1][idx] = data

        for epoch_list in checkpoint['weighted metric']:
            all_checkpoints['weighted metric'].append([[]]*len(checkpoint['weighted metric'][0]))
            for idx, data in enumerate(epoch_list):
                all_checkpoints['weighted metric'][-1][idx] = data
        
        for epoch_list in checkpoint['macro metric']:
            all_checkpoints['macro metric'].append([[]]*len(checkpoint['macro metric'][0]))
            for idx, data in enumerate(epoch_list):
                all_checkpoints['macro metric'][-1][idx] = data
        
        for epoch_list in checkpoint['numerical metric']:
            all_checkpoints['numerical metric'].append([[]]*len(checkpoint['numerical metric'][0]))
            for idx, data in enumerate(epoch_list):
                all_checkpoints['numerical metric'][-1][idx] = data

        if i == max:
            all_checkpoints['current epoch'] = (0, checkpoint['current epoch'][1])

    
    return all_checkpoints

  