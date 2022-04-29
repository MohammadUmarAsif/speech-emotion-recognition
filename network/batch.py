import torch
import numpy as np
from helper.common import shuffle, to_tensor, load_image
from helper.network import normalize_tensor


def apply_transforms(img: list, img_size: tuple, mean: list, std: list, color_map: str) -> torch.Tensor:
    '''
    Applies transforms to an image

    param: img = image data
    param: img_size = size of image
    param: mean = mean of each channel
    param: std = standard deviation of each channel
    param: color map = color map of image

    return: transformed image tensor
    '''

    img = img.resize(img_size)
    img_tensor = to_tensor(img)
    img_tensor = normalize_tensor(img_tensor, mean, std, color_map)
    
    return img_tensor


def fill_last_batch(last_batch: list, batch_size: int, data: list, size: int) -> list:
    '''
    Fills the last batch with random samples

    param: last_batch = last batch of samples
    param: batch_size = size of each batch
    param: data = shuffled samples
    param: size = length of shuffled samples

    return last batch of samples
    '''

    indices = torch.randperm(size).tolist()
    num_missing = batch_size - (size % batch_size)

    indices = indices[:num_missing]
    
    for i in range(num_missing):
        idx = indices[i]
        last_batch.append(data[idx])
    
    return last_batch


def get_current_batch(batches: list, batch_size: int, batch_count: int, shuffled_data: list) -> tuple:
    '''
    Gets a batch 

    param: batches = batches of samples
    param: batch_size = size of each batch
    param: batch_count = current batch index
    param: shuffled_data = shuffled samples

    return batch of samples
    '''

    if batch_count == len(batches)-1:
        return fill_last_batch(list(batches[-1]), batch_size, shuffled_data, len(shuffled_data))
    
    else:
        return batches[batch_count]


def load_iemocap_batch(root: str, split_type: str, label_data: dict, class_data: dict, batch: list, batch_count: list,
                        batch_type: int, transforms: list, scaled_vad = None, scaling_type = None) -> tuple:
    '''
    Loads the data of iemocap samples in a batch

    param: root = root folder name
    param: split_type = type of split
    param: label_data = labels of samples
    param: class_data = mappings of labels to integers
    param: batch = batch of samples
    param: batch_count = current batch index
    param: transforms = image transformations
    param: scaled_vad = scaled numerical values
    param: scaling_type = type of scaling

    return: image tensors, label tensors, sample names
    '''
    
    batch_tensors = []

    # Getting batch content
    for key in batch:
        file_name = key[:-5]
        value = label_data[file_name]
        label = value[0]
        
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{key}.png'

        img = load_image(path, transforms[3])
        transformed_img = apply_transforms(img, transforms[0], transforms[1], transforms[2], transforms[3])

        emotion_int = class_data[value[0]]
        label_list = [emotion_int]

        if scaled_vad:
            gender_int = class_data[value[1]]
            label_list.extend([gender_int])

            scaled_value = scaled_vad[file_name]
            if scaling_type == 'min-max':
                min_max = (scaled_value[0], scaled_value[2], scaled_value[4])
                label_list.extend(list(min_max))
            elif scaling_type == 'standard':
                standard = (scaled_value[1], scaled_value[3], scaled_value[5])
                label_list.extend(list(standard))
        
        tensor_labels = torch.as_tensor(label_list, dtype=torch.float32)

        batch_tensors.append((transformed_img, tensor_labels, key))
    
    if batch_count:
        batch_count[batch_type] += 1
    tensor_inputs, tensor_labels, names = [],[],[]
    
    # Organizing batch content
    for i in range(len(batch_tensors)):
        tensor_inputs.append(batch_tensors[i][0])
        tensor_labels.append(batch_tensors[i][1])
        names.append(batch_tensors[i][2])
    
    return (torch.stack(tensor_inputs), torch.stack(tensor_labels), names)


def create_batches(data: list, batch_size: int) -> tuple:
    '''
    Creates batches of fixed size from shuffled samples
    
    param: data = samples
    param: batch_size = size of each batch
    
    return: batches and shuffled data
    '''
    
    shuffled_data = shuffle(data)
    
    indices = torch.randperm(len(shuffled_data)).tolist()
    num_batches = int(np.ceil(len(shuffled_data)/batch_size))
    batches = []
    
    for i in range(num_batches):
        start = i*batch_size
        stop = (i+1)*batch_size
        batch_idx = indices[start:stop]
        
        batches.append([])

        for idx in batch_idx:
            batches[i].append(shuffled_data[idx])
            
    return (batches, shuffled_data)
        
    
def get_augmentations(data: list, num_augs: list) -> list:
    '''
    Gets required augmentations of the data
    
    param: data = samples
    param: num_aug = number of augmentations
    
    return: names of augmented data files
    '''
    
    aug_data = []

    for key in data:     
        for i in num_augs:
            aug_data.append(f'{key}_aug{str(i)}')
    
    return aug_data


def get_labels(labels: torch.Tensor, label_idx: int, label_type: str) -> torch.Tensor:
    '''
    Gets all labels of a particular task

    param: label = all labels of all tasks
    param: label_idx = index of particular task label
    param: label_type = type of label

    return: all labels of a particular task
    '''

    label_list = []
    
    if label_type == 'classification':
        dtype = torch.long
        
        for label in labels:
            label_list.append(int(label[label_idx]))

        tensor = torch.as_tensor(label_list, dtype=dtype)
        return tensor

    elif label_type == 'regression':
        dtype = torch.float32
        
        for label in labels:
            label_list.append(label[label_idx])
   
        tensor = torch.as_tensor(label_list, dtype=dtype)
        return tensor.view(-1, 1)


def get_batches(batch_size: int, split_data: dict, split_types: list, num_augs: list) -> tuple:
    '''
    Gets batches based on required augmentations

    param: batch_size = size of each batch
    param: split_data = split samples
    param: split_types = types of split
    param: num_augs = number of augmentations for each split

    return: batches and shuffled data
    '''
    
    data_aug = []
    
    for idx, split in enumerate(split_types):
        data_aug.append(get_augmentations(split_data[split], num_augs = num_augs[idx]))

    data_batches, shuffled_data = [], []
    for data in data_aug:
        batches, shuffled = create_batches(data, batch_size)
        data_batches.append(batches)
        shuffled_data.append(shuffled)
    
    return (data_batches, shuffled_data)


def load_ravdess_batch(root: str, split_type: str, label_data: dict, class_data: dict, batch: list, transforms: list) -> tuple:
    '''
    Loads the data of ravdess samples in a batch

    param: root = root folder name
    param: split_type = type of split
    param: label_data = labels of samples
    param: class_data = mappings of labels to integers
    param: batch = batch of samples
    param: transforms = image transformations

    return: image tensors, label tensors, sample names
    '''
    
    batch_tensors = []

    # Getting batch content
    for key in batch:
        file_name = key[:-5]
        value = label_data[file_name]
        label = value[0]
        
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{key}.png'

        img = load_image(path, transforms[3])
        transformed_img = apply_transforms(img, transforms[0], transforms[1], transforms[2], transforms[3])

        emotion_int = class_data[value[0]]
        label_list = [emotion_int]

        gender_int = class_data[value[1]]
        inten_int = class_data[value[2]]
        label_list.extend([gender_int, inten_int])
        
        tensor_labels = torch.as_tensor(label_list, dtype=torch.float32)

        batch_tensors.append((transformed_img, tensor_labels, key))
    
    tensor_inputs, tensor_labels, names = [],[],[]
    
    # Organizing batch content
    for i in range(len(batch_tensors)):
        tensor_inputs.append(batch_tensors[i][0])
        tensor_labels.append(batch_tensors[i][1])
        names.append(batch_tensors[i][2])
    
    return (torch.stack(tensor_inputs), torch.stack(tensor_labels), names)
