import json
import torch
import torch.nn as nn
import numpy as np
from torchmetrics import Accuracy, F1Score, MeanSquaredError, ConfusionMatrix
from helper.common import to_tensor, load_image


def get_channel_stats(root: str, split_data: dict, label_data: dict, color_map: str) -> tuple:
    '''
    Calculates the mean and standard deviation across each channel for all images
    
    param: root = root folder name
    param: split_data = split samples
    param: label_data = labels of samples
    param: color_map = color map of image

    return: mean and standard deviation
    '''
    
    channels = 1 if color_map == 'L' else 3
    mean = np.zeros(channels)
    std = np.zeros(channels)

    total_samples = 0

    for split_type, keys in split_data.items():
        for key in keys:
            label = label_data[key][0]
            
            if split_type == 'valid':
                split_type = 'train'
            
            path = f'{root}/{split_type.capitalize()}/{label}/{key}/{key}_aug1.png'
            
            img = load_image(path, color_map)
            img = to_tensor(img)

            for i in range(channels):
                mean[i] += img[i].mean()
                std[i] += img[i].std()
        
        total_samples += len(keys)
    
    for i in range(channels):
        mean[i] = mean[i]/total_samples
        std[i] = std[i]/total_samples

    return (mean, std)
        

def inverse_scaler(preds, labels, stats, scaling_type) -> None:
    '''
    Inverses the scaling
    
    param: preds = predictions of samples
    param: labels = labels of samples
    param: stats = statistical values
    param: scaling_type = type of scaling
    
    return: None
    '''
    
    if scaling_type == 'min-max':
        for idx in [2, 3, 4]:
            preds[idx] = preds[idx]*stats['diff'][idx-2] + stats['min'][idx-2]
        
            if labels:
                labels[idx] = labels[idx]*stats['diff'][idx-2] + stats['min'][idx-2]

    elif scaling_type == 'standard':
        for idx in [2, 3, 4]:
            preds[idx] = preds[idx]*stats['std'][idx-2] + stats['mean'][idx-2]

            if labels:
                labels[idx] = labels[idx]*stats['std'][idx-2] + stats['mean'][idx-2]

        

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


def get_emotion_num(name: str, class_data: dict) -> int:
    '''
    Gets number of emotion classes

    param: name = name of dataset
    param: class_data = mappings of class to integers

    return: number of classes
    '''
    
    if name == 'iemocap':
        return len(class_data) - 2
    elif name == 'ravdess':
        return len(class_data) - 4


def normalize_tensor(tensor: torch.Tensor, mean: list, std: list, color_map: str) -> torch.Tensor:
    '''
    Normalizes a tensor

    param: tensor = input tensor 
    param: mean = mean of each channel
    param: std = standard deviation of each channel
    param: color_map = color map of image

    return: normalized tensor
    '''
    
    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype)
    std = torch.as_tensor(std, dtype=dtype)
    
    if color_map == 'RGB':
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    
    tensor.sub_(mean).div_(std)

    return tensor


def get_classification_metric(average_type: str, num_emotion = 0, num_gender = 0, num_intensity = None, top_k = 1) -> list:
    '''
    Obtains the metrics requried for classification tasks

    param: average_type = type of averaging
    param: num_emotion = number of emotion outputs
    param: num_gender = number of gender outputs

    return: metrics
    '''
    
    metrics = []

    if num_emotion:
        metrics.append(Accuracy(num_classes = num_emotion, average = average_type, top_k = top_k))
        metrics.append(F1Score(num_classes = num_emotion, average = average_type, top_k = top_k))
    
    if num_gender:
        metrics.append(Accuracy(num_classes = num_gender, average = average_type, top_k = 1))
        metrics.append(F1Score(num_classes = num_gender, average = average_type, top_k = 1))

    if num_intensity:
        metrics.append(Accuracy(num_classes = num_intensity, average = average_type, top_k = 1))
        metrics.append(F1Score(num_classes = num_intensity, average = average_type, top_k = 1))
    
    return metrics


def get_regression_metric(num_metrics: int) -> list:
    '''
    Obtains the metric requried for regression tasks

    param: num_metrics = number of metrics

    return: metrics
    '''

    metrics = []

    for i in range(num_metrics):
        metrics.append(MeanSquaredError(squared = False))
    
    return metrics


def get_criterion(task_type: tuple) -> list:
    '''
    Obtains the criterion (loss functions) for all tasks

    param: task_type = type of each task

    return: criterions
    '''

    criterion = []

    for task in task_type:
        if task == 'classification':
            criterion.append(nn.NLLLoss())
        elif task == 'regression':
            criterion.append(nn.MSELoss())
    
    return criterion


def get_confusion_matrix(num_emotion: int, num_gender: int, num_intensity = None, name = 'iemocap') -> list:
    '''
    Obtains the confusion matrices required for classification

    param: num_emotion = number of emotion outputs
    param: num_gender = number of gender outputs

    return: confusion matrices
    '''

    if name == 'iemocap':
        return [ConfusionMatrix(num_classes = num_emotion), ConfusionMatrix(num_classes = num_gender)]
    elif name == 'ravdess':
        return [ConfusionMatrix(num_classes = num_emotion), ConfusionMatrix(num_classes = num_gender),
                ConfusionMatrix(num_classes = num_intensity)]


def ensemble_preds(preds: list, pred_labels: list, name = 'iemocap') -> tuple:
    '''
    Combines the predictions for test-time augmentation evaluation

    param: preds = predictions of model
    param: pred_labels = labels of predictions
    param: name = name of dataset

    return: combined predictions
    '''
    
    num_samples = len(preds[0])
    if num_samples == 1:
      return (preds, pred_labels)

    for i in range(0, 2):
        preds[i] = torch.exp(preds[i])     
        preds[i] = torch.sum(preds[i], dim = 0)/num_samples
        preds[i] = torch.log(preds[i])
        preds[i] = torch.as_tensor([preds[i].tolist()])

    for i in range(2, 5):
        preds[i] = preds[i].sum(dim = 0)/num_samples
        preds[i] = torch.as_tensor([[preds[i]]])

    if name == 'ravdess':
      for i in range(0, 3):
          pred_labels[i] = torch.as_tensor([pred_labels[i][0]])
    
    elif name == 'iemocap':
      for i in range(0, 2):
          pred_labels[i] = torch.as_tensor([pred_labels[i][0]])
      
      for i in range(2, 5):
          pred_labels[i] = torch.as_tensor([[pred_labels[i][0]]])

    return (preds, pred_labels)

