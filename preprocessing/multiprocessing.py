import multiprocessing
from ipywidgets import IntProgress, Layout
from IPython.display import display
from helper.common import load_audio_data
from augmentation import augment_files, augment_spectrogram
from spectrogram import audio_to_spec


def multi_audio_aug(root: str, split_type: str, factors: list, num_subsets: int) -> None:
    '''
    Performs audio augmentation through multiprocessing

    param: root = root folder name
    param: split_type = type of split
    param: factors = augmentation factors
    param: num_subsets = number of subsets

    return: None
    '''

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = num_subsets, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'
    display(progress_bar)
    
    for idx in range(num_subsets):
        audio_subset = load_audio_data(root, split_type, f'temp{idx}')
        save_parameters = (root, split_type, f'aug{idx}')
        process = multiprocessing.Process(target = augment_files, args = (audio_subset, factors, save_parameters))
        process.start()
        process.join()
        
        progress_bar.value = idx + 1
        if progress_bar.value == num_subsets:
            progress_bar.description = f'Augmentation Completed'
        else:
            progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'


def multi_audio_to_spec(root: str, label_data: dict, split_type: str, spec_params: tuple, num_subsets: int) -> None:
    '''
    Performs conversion of audio to spectrograms through multiprocessing

    param: root = root folder name
    param: label_data = labels of samples
    param: split_type = type of split
    param: spec_params = parameters for spectrograms
    param: num_subsets = number of subsets

    return: None
    '''

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = num_subsets, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'
    display(progress_bar)

    for idx in range(num_subsets):
        audio_subset = load_audio_data(root, split_type, f'aug{idx}')
        process = multiprocessing.Process(target = audio_to_spec, args = (root, audio_subset, split_type, label_data, spec_params))
        process.start()
        process.join()

        progress_bar.value = idx + 1
        if progress_bar.value == num_subsets:
            progress_bar.description = f'Conversion Completed'
        else:
            progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'


def multi_spec_aug(root: str, label_data: dict, split_type: str, width: list, num_subsets: int) -> None:
    '''
    Performs spectrogram augmentation through multiprocessing

    param: root = root folder name
    param: label_data = labels of samples
    param: split_type = type of split
    param: width = lower and upper width of mask
    param: num_subsets = number of subsets

    return: None
    '''

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = num_subsets, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'
    display(progress_bar)

    for idx in range(num_subsets):
        audio_subset = load_audio_data(root, split_type, f'aug{idx}')
        process = multiprocessing.Process(target = augment_spectrogram, args = (root, audio_subset, split_type, label_data, width))
        process.start()
        process.join()

        progress_bar.value = idx + 1
        if progress_bar.value == num_subsets:
            progress_bar.description = f'Augmentation Completed'
        else:
            progress_bar.description = f'Subset - {progress_bar.value} / {num_subsets}'

