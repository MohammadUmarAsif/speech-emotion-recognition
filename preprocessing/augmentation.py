import librosa
import torch
import numpy as np
from spectrogram import save_from_img
from helper.preprocessing import save_audio_data
from helper.common import to_tensor, load_image


def add_white_noise(x: list, noise_factor: float) -> list:
    '''
    Adds white noise to input signal

    param: x = input signal
    param: noise_factor = % of noise to be added

    return: augmented signal
    '''
    
    noise = np.random.normal(0, x.std(), x.size)
    augmented_signal = x + noise*noise_factor
    
    return augmented_signal


def time_stretch(x: list, stretch_rate: float) -> list:
    '''
    Stretches input signal by a factor

    param: x = input signal
    param: stretch_rate = amount by which signal is stretched

    return: augmented signal
    '''

    return librosa.effects.time_stretch(x, stretch_rate)


def pitch_shift(x: list, sample_rate: int, num_semitones: int) -> list:
    '''
    Shifts input signal pitch by number of semitones

    param: x = input signal
    param: sample_rate = sample rate of signal
    param: num_semitones = number of semitones

    return: augmented signal
    '''

    return librosa.effects.pitch_shift(x, sample_rate, num_semitones)


def augment_audio(x: list, sample_rate: int, factors: list) -> list:
    '''
    Performs audio augmentations

    param: x = input signal
    param: sample_rate = sample rate of signal
    param: factors = augmentation factors
    
    return: noise, time, and pitch augmented signals
    '''
    
    noise_aug = add_white_noise(x, factors[0])
    
    time_factor = np.random.choice([factors[1][0], factors[1][1]])
    time_aug = time_stretch(x, time_factor)
    
    pitch_factor = np.random.choice([factors[2][0], factors[2][1]])
    pitch_aug = pitch_shift(x, sample_rate, pitch_factor)

    return [noise_aug, time_aug, pitch_aug]


def augment_files(audio_data: dict, factors: list, save_parameters: tuple) -> None:
    '''
    Augments audio data

    param: audio_data = audio of samples
    param: factors = augmentation factors
    param: save_parameters = parameters for saving function

    return: None
    '''

    for key, value in audio_data.items():        
        sample_rate, x = value[0], value[1]
        
        augmented_audio = augment_audio(x, sample_rate, factors)
        value.extend(augmented_audio)
        
        audio_data[key] = value

    root, split_type, name = save_parameters
    save_audio_data(root, audio_data, split_type, name)


def freq_mask(img: torch.Tensor, lower: int, upper: int) -> torch.Tensor:
    '''
    Applies a frequency mask to the spectrogram image

    param: img = tensor image
    param: lower = lower limit of mask width
    param: upper = upper limit of mask width

    return: augmented spectrogram tensor
    '''
    
    img_tensor = to_tensor(img)

    freq_max = img_tensor.shape[1]
    f = np.random.uniform(size = (), low = lower, high = upper)
    f0 = np.random.uniform(low = 0, high = freq_max - f,)
    
    indices = torch.reshape(torch.arange(freq_max), (-1, 1))
    condition = torch.logical_and(torch.greater_equal(indices, f0), torch.less(indices, f0 + f))
    c = torch.as_tensor(0, dtype = img_tensor.dtype)
    
    return torch.where(condition, c, img_tensor)


def time_mask(img: torch.Tensor, lower: int, upper: int) -> torch.Tensor:
    '''
    Applies a time mask to the spectrogram image

    param: img = tensor image
    param: lower = lower limit of mask width
    param: upper = upper limit of mask width

    return: augmented spectrogram tensor
    '''

    img_tensor = to_tensor(img)

    time_max = img_tensor.shape[2]
    t = np.random.uniform(size = (), low = lower, high = upper)
    t0 = np.random.uniform(low = 0, high = time_max - t,)
    
    indices = torch.reshape(torch.arange(time_max), (1, -1))
    condition = torch.logical_and(torch.greater_equal(indices, t0), torch.less(indices, t0 + t))
    c = torch.as_tensor(0, dtype = img_tensor.dtype)

    return torch.where(condition, c, img_tensor)


def augment_spectrogram(root: str, audio_data: dict, split_type: str, label_data: dict, width: list) -> None:   
    '''
    Augments and saves the masked spectrograms
    
    param: root = root folder name
    param: audio_data = audio of samples
    param: split_type = type of split
    param: label_data = labels of samples
    param: width = lower and upper width of mask

    return: None
    '''

    for key in audio_data.keys():
        file_name = key
        label = label_data[key][0]
        
        source_path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug1.png'
        img = load_image(source_path, 'RGB')
        
        freq_img = freq_mask(img, width[0], width[1])
        time_img = time_mask(img, width[0], width[1])
        
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug5.png'
        save_from_img(freq_img, path)               
        
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug6.png'             
        save_from_img(time_img, path)     

