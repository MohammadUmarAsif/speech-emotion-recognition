import os
import torch
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def audio_to_spec(root: str, audio_data: dict, split_type: str, label_data: dict, spec_params: tuple) -> None:
    '''
    Converts audio data into spectrograms
    
    param: root = root folder name
    param: audio_data = audio of samples
    param: split_type = type of split
    param: label_data = labels of samples
    param: spec_params = parameters for spectrograms

    return: None
    '''

    for key, value in audio_data.items():
        file_name = key
        label = label_data[key][0]
        sample_rate = value[0]

        os.mkdir(f'{root}/{split_type.capitalize()}/{label}/{file_name}')  

        for i in range(1, len(value)):
            path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug{str(i)}.png'
            save_from_audio(value[i], sample_rate, path, spec_params)


def save_from_audio(x: list, sample_rate: int, path: str, spec_params: tuple) -> None:
    '''
    Saves spectrograms of 256x256 pixels from audio

    param: x = input signal
    param: sample_rate = sample rate of signal
    param: path = path of saved file
    param: spec_params = parameters for spectrograms

    return: None
    '''

    log_mel = create_spectrogram(x, sample_rate, spec_params)
    
    # Remove margins from image before saving
    plt.figure(figsize = [3.56, 3.56])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0,0)

    quadmesh = librosa.display.specshow(log_mel, sr = sample_rate)
    quadmesh.figure.savefig(path, bbox_inches = 'tight', pad_inches = 0)
    plt.close(quadmesh.figure)


def save_from_img(img: torch.Tensor, path: str) -> None:
    '''
    Saves spectrograms of 256x256 pixels from images

    param: img = image tensor
    param: path = path of saved file

    return: None
    '''

    img = img.permute((1, 2, 0))
    img = img.numpy()
    
    # Remove margins from image before saving
    plt.figure(figsize = [3.56,3.56])
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.margins(0,0)
    
    plt.imshow(img)
    plt.savefig(path, bbox_inches = 'tight', pad_inches = 0)
    plt.close(plt.gcf())


def create_spectrogram(x, sample_rate, spec_params) -> np.ndarray:
    '''
    Creates a spectrogram based on parameters
    
    param: x = input signal
    param: sample_rate = sample rate of signal
    param: spec_params = parameters for spectrograms
    
    return: spectrogram
    '''

    n_fft, hop_length, n_mels = spec_params

    fft_windows = librosa.stft(x, n_fft = n_fft, hop_length = hop_length)
    magnitude = np.abs(fft_windows)**2
    mel = librosa.filters.mel(sr = sample_rate, n_fft = n_fft, n_mels = n_mels)
    mel = mel.dot(magnitude)    
    log_mel = librosa.power_to_db(mel, ref = np.max)

    return log_mel

