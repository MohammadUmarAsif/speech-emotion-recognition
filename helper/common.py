import torch
import json
import numpy as np
from PIL import Image


def load_data(root: str, name: str) -> dict:
    '''
    Loads json data into dictionary

    param: root = root folder name
    param: name = file name

    return: data
    '''
    
    path = f'{root}/{name}.json'
    data = None
    
    with open(path, 'r') as file:
        data = json.load(file)
    
    return data


def shuffle(data: list) -> list:
    '''
    Shuffles elements of a list

    param: data = data to be shuffled

    return: shuffled data
    '''
    
    shuffled_data = []
    indices = torch.randperm(len(data)).tolist()

    for idx in indices:
        shuffled_data.append(data[idx])
    
    return shuffled_data


def load_audio_data(root: str, split_type: str, name: str) -> dict:
    '''
    Loads npy data into dictionary

    param: root = root folder name
    param: split_type = type of split
    param: name = name of subset

    return: audio of samples
    '''

    path = f'{root}/Audio/{split_type.capitalize()}/{name}.npy'
    audio_data = np.load(path, allow_pickle = True)
        
    return audio_data.item()


def save_data(root: str, data: dict, name: str) -> None: 
    '''
    Saves dictionary in json format

    param: root = root folder name
    param: data = data to be saved
    param: name = file name

    return: None
    '''

    path = f'{root}/{name}.json'
    with open(path, 'w') as file:
        json.dump(data, file)


def load_image(path: str, color_map: str) -> list:
    '''
    Loads RGB image

    param: path = path of image file
    param: color_map = color map of image

    return: image data
    '''

    with open(path, 'rb') as file:
        img = Image.open(file)
        img = img.convert(color_map)
        return img


def to_tensor(img: list) -> torch.Tensor:
    '''
    Converts PIL image to tensor

    param: img = image data

    return: image tensor
    '''

    img_tensor = torch.from_numpy(np.array(img, np.uint8, copy=True))
    img_tensor = img_tensor.view(img.size[1], img.size[0], len(img.getbands()))
    img_tensor = img_tensor.permute((2, 0, 1))

    return img_tensor.to(dtype=torch.float32).div(255)

