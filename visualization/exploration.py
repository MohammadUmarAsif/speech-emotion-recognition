import torch
import librosa
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import numpy as np
import IPython.display as ipd
from PIL import Image
from helper.common import load_audio_data, load_image
from helper.visualization import hide_empty


COLORS = {'primary': '#003d7a', 'secondary': '#ff8400'}
HSPACE = 0.4
FIG_WIDTH = 18
FIG_HEIGHT = 4


def plot_percent(number_df: pd.DataFrame, class_labels: list, percent_df: pd.DataFrame) -> None:
    '''
    Plots distributions of values in a column with percentages

    param: number_df = for amount of values
    param: class_labels = labels of classes
    param: percent_df = for percentage of values

    return: None
    '''

    column = 'Emotion'

    plots = sb.countplot(x = number_df[column], order = class_labels, palette = [COLORS['primary']]);
    plt.ylabel('Number of samples')
    plt.title(f'Distribution of Samples ({column})')
    
    max_bar = max([bar.get_height() for bar in plots.patches])

    for idx, bar in enumerate(plots.patches):    
        total = 0   
        
        if type(percent_df) != type(None):
            total = percent_df[percent_df[column] == class_labels[idx]].shape[0]
        else:
            total = number_df.shape[0]

        plots.annotate(f'{bar.get_height():.0f}', (bar.get_x()+bar.get_width()/2, 
                        bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
        plots.annotate(f'({(bar.get_height()/total)*100:.2f}%)', (bar.get_x()+bar.get_width()/2, 
                        max_bar + 0.1*max_bar), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
    
    plt.ylim(0, max_bar + 0.2*max_bar)


def plot_categorical_values(df: pd.DataFrame, class_labels: list, plot_data: tuple) -> None:
    '''
    Plots distribution of categorical values using bars

    param: df = dataframe
    param: class_labels = labels of classes
    param: plot_data = data for plotting

    return: None
    '''

    distributions, legend, orders = plot_data
    
    num_cols = 1
    num_rows = int(np.ceil((len(distributions)+1)/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)

    plot_idx = 1

    plt.subplot(num_rows, num_cols, plot_idx)
    plot_percent(df, class_labels, None)
    
    plot_idx += 1
    
    for key, hue in distributions:
        plt.subplot(num_rows, num_cols, plot_idx)
        plots = sb.countplot(x = df[key], palette = [COLORS['primary'], COLORS['secondary']], order = orders[key], hue = df[hue]);
    
        max_bar = 0
        for bar in plots.patches:
            if bar.get_height() > max_bar:
                max_bar = bar.get_height()
            
            plots.annotate(f'{bar.get_height():.0f}', (bar.get_x()+bar.get_width()/2, bar.get_height()), ha = 'center', va = 'center', 
                            size = 10, xytext = (0,8), textcoords = 'offset points')
        
        plt.ylim(0, max_bar + 0.1*max_bar)
        
        plt.ylabel('Number of samples')
        plt.title(f'Distribution of Samples ({key} and {hue})')
        plt.legend(bbox_to_anchor = (1.01, 1), loc = 'upper left', borderaxespad = 0, title = hue, labels = legend[hue])

        plot_idx += 1
    
    hide_empty(plot_idx, num_rows, num_cols)


def plot_numerical_values(df: pd.DataFrame, bin_size: float, columns: list) -> None:
    '''
    Plots distribution of numerical values using histograms

    param: df = dataframe
    param: bin_size = size of bins
    param: columns = numerical value columns

    return: None
    '''
    
    num_rows = 1
    num_cols = 3
    plt.subplots(num_rows, num_cols, figsize=(FIG_WIDTH, FIG_HEIGHT*num_rows))
    plt.subplots_adjust(hspace = HSPACE)
    plot_idx = 1

    for col in columns:
        plt.subplot(num_rows, num_cols, plot_idx)
        bins = [i*bin_size for i in range(int((1/bin_size)*(np.floor(df[col].min()))) - 1, int((1/bin_size)*(np.ceil(df[col].max()))) + 1)]
        plt.hist(x = df[col], bins = bins, color = COLORS['primary']); 
        plt.xticks(bins);
        plt.title(col)

        plot_idx += 1
    
    hide_empty(plot_idx, num_rows, num_cols)


def plot_iemocap_distributions(df: pd.DataFrame, class_labels: list) -> None:
    '''
    Plots distributions of iemocap samples

    param: df = dataframe
    param: class_labels = labels of classes

    return: None
    '''

    distributions = [['Emotion', 'Gender']]
    legend = {'Gender': ['Male', 'Female']}
    orders = {'Emotion': class_labels}
    
    plot_data = (distributions, legend, orders)
    
    plot_categorical_values(df, class_labels, plot_data)
    plot_numerical_values(df, bin_size = 0.5, columns = ['Valence', 'Activation', 'Dominance'])


def plot_ravdess_distributions(df: pd.DataFrame, class_labels: list) -> None:
    '''
    Plots distributions of ravdess samples

    param: df = dataframe
    param: class_labels = labels of classes

    return: None
    '''

    distributions = [['Emotion', 'Gender'], ['Emotion', 'Intensity'], ['Gender', 'Intensity']]
    legend = {'Gender': ['Male', 'Female'], 'Intensity': ['Normal', 'Strong']}
    orders = {'Emotion': class_labels, 'Gender': ['Male', 'Female']}

    plot_data = (distributions, legend, orders)
    
    plot_categorical_values(df, class_labels, plot_data)


def plot_audio(label_df: pd.DataFrame, audio_data: dict, name: str, label: str, num_samples: int, playable: bool) -> None:
    '''
    Plots audio waveforms for a given label

    param: label_df = labels of samples
    param: audio_data = audio of samples
    param: name = name of dataset
    param: label = class label
    param: num_samples = number of samples to plot
    param: playable = should playable audio be shown

    return: None
    '''

    num_cols = 3
    num_rows = int(np.ceil(num_samples/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)

    plot_idx = 1

    while plot_idx <= num_samples:
        gender = np.random.choice(['Male', 'Female'])
        
        row = label_df[(label_df.Emotion == label) & (label_df.Gender == gender)].sample()
        file_name = row.values[0][0]
        title = f'{plot_idx}: {file_name} | {label} | {gender}'

        if name == 'ravdess':
            intensity = np.random.choice(['Normal', 'Strong'])
            row = label_df[(label_df.Emotion == label) & (label_df.Gender == gender) & (label_df.Intensity == intensity)].sample()
            file_name = row.values[0][0]
            title = f'{plot_idx}: {file_name} | {label} | {gender} | {intensity}'
        
        sample_rate, x = audio_data[file_name]

        if playable:
            print(title)
            ipd.display(ipd.Audio(x, rate = sample_rate))
        
        plt.subplot(num_rows, num_cols, plot_idx)
        librosa.display.waveshow(x, sr = sample_rate, color = COLORS['primary']);
        plt.title(title)

        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)
    

def plot_trimmed_audio(detail_df: pd.DataFrame, audio_data: dict, trimmed_audio_data: dict, num_samples: int, playable: bool) -> None:
    '''
    Plots original and trimmed audio waveforms of random samples

    param: detail_df = details of samples
    param: audio_data = audio of samples
    param: trimmed_audio_data = trimmed audio of samples
    param: num_samples = number of samples to plot
    param: playable = should playable audio be shown

    return: None
    '''
    
    num_cols = 2
    num_rows = num_samples
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)

    audio_idx = 1
    plot_idx = 1

    while plot_idx <= num_samples*2:
        row = detail_df.sample()
        file_name = row.values[0][0]
        duration = row.values[0][2]
        
        sample_rate, x = audio_data[file_name]
        title = f'{audio_idx}: {file_name} | Original | {duration} sec'
        
        if playable:
            print(title)
            ipd.display(ipd.Audio(x, rate = sample_rate))
            
        
        plt.subplot(num_rows, num_cols, plot_idx)
        librosa.display.waveshow(x, sr = sample_rate, color = COLORS['primary']);
        plt.title(title)

        duration = row.values[0][1]
        sample_rate, x = trimmed_audio_data[file_name]
        title = f'{audio_idx}: {file_name} | Trimmed | {duration} sec'

        if playable:
            print(title)
            ipd.display(ipd.Audio(x, rate = sample_rate))
            
        plt.subplot(num_rows, num_cols, plot_idx+1)
        librosa.display.waveshow(x, sr = sample_rate, color = COLORS['secondary']);
        plt.title(title)

        audio_idx += 1
        plot_idx += 2


def plot_duration(detail_df: pd.DataFrame, bin_size: float) -> None:
    '''
    Plots duration of samples

    param: detail_df = details of samples
    param: bin_size = size of bins

    return: None
    '''

    max = detail_df['Original Duration'].max()
    min = detail_df['Original Duration'].min()
    mean = detail_df['Original Duration'].mean()

    print('-------------Original Duration-------------')
    print(f'Maximum: {max:.02f} sec')
    print(f'Minimum: {min:.02f} sec')
    print(f'Average: {mean:.02f} sec')

    max = detail_df['Trimmed Duration'].max()
    min = detail_df['Trimmed Duration'].min()
    mean = detail_df['Trimmed Duration'].mean()

    print('\n-------------Trimmed Duration-------------')
    print(f'Maximum: {max:.02f} sec')
    print(f'Minimum: {min:.02f} sec')
    print(f'Average: {mean:.02f} sec')

    plt.subplots(1, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    plt.subplot(1,2,1)
    bins = [i*bin_size for i in range(0, int((1/bin_size)*(np.ceil(detail_df['Original Duration'].max()))) + 1)]
    plt.hist(x = detail_df['Original Duration'], bins = bins, color = COLORS['primary']);
    plt.xticks(bins);
    plt.title('Original Duration Distribution')
    
    plt.subplot(1,2,2)
    bins = [i*bin_size for i in range(0, int((1/bin_size)*(np.ceil(detail_df['Trimmed Duration'].max()))) + 1)]
    plt.hist(x = detail_df['Trimmed Duration'], bins = bins, color = COLORS['secondary']);
    plt.xticks(bins);
    plt.title('Trimmed Duration Distribution')


def plot_split(split_data: dict, label_df: pd.DataFrame, label_data: dict, class_labels: list) -> None:
    '''
    Plots the distribution of labels across the split data

    param: split_data = split samples
    param: label_df = labels of samples
    param: label_data = labels of samples
    param: class_labels = labels of classes

    return: None
    '''

    num_cols = 2
    num_rows = int(np.ceil(len(split_data.keys())/num_cols))
    
    plot_idx = 1
    split_labels = []
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)

    for i, item in enumerate(split_data.items()):
        key, value = item
        split_labels.append([])
        
        for name in value:
            label = label_data[name][0]
            split_labels[i].append(label)
        
        plt.subplot(num_rows, num_cols, plot_idx)
        df = pd.DataFrame(split_labels[i], columns=['Emotion'])
        
        plot_percent(df, class_labels, label_df)
        plt.title(f'Amount & Percentage of each emotion in {key.capitalize()}')
        
        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)


def plot_audio_aug(root: str, split_type: str, num_subsets: int, playable: bool) -> None:
    '''
    Plots augmentations of a random sample of audio

    param: root = root folder name
    param: split_type = type of split
    param: num_subsets = number of subsets
    param: playable = should playable audio be shown

    return: None
    '''
    
    num_cols = 2
    num_rows = 2
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)

    subset_idx = np.random.choice([i for i in range(num_subsets)])
    subset = load_audio_data(root, split_type, f'aug{subset_idx}')
    file_name = np.random.choice(list(subset.keys()))
    
    sample_rate, x1, x2, x3, x4 = subset[file_name]
    audio = {'Original': x1, 'Noise Addition': x2, 'Time Stretch': x3, 'Pitch Shift': x4}
    
    for idx, item in enumerate(audio.items()):
        aug, x = item
        title = f'{idx+1}: {file_name} | {aug}'
        
        if playable:
            print(title)
            ipd.display(ipd.Audio(x, rate = sample_rate))

        plt.subplot(num_rows, num_cols, idx+1)
        librosa.display.waveshow(x, sr = sample_rate, color=COLORS['primary']);
        plt.title(title)


def plot_spectrogram(root: str, split_data: dict, split_type: str, label_data: dict, class_labels: list, name: str) -> None:
    '''
    Plots a spectrogram of each label in the data

    param: root = root folder name
    param: split_data = split samples
    param: split_type = type of split
    param: label_data = labels of samples
    param: class_labels = labels of classes
    param: name = name of dataset

    return: None
    '''
    
    ordered = {}
    for label in class_labels:
        ordered[label] = []
    
    # Organize samples by labels
    for name in split_data[split_type]:
        label = label_data[name][0]
        ordered[label].append(name)
    
    num_cols = 3
    num_rows = int(np.ceil(len(class_labels)/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)
    
    plot_idx = 1

    for label in class_labels:
        file_name = np.random.choice(ordered[label])
        gender = label_data[file_name][1]
        title = f'{file_name} | {label} | {gender}'

        if name == 'ravdess':
            intensity = label_data[file_name][2]
            title = f'{file_name} | {label} | {gender} | {intensity}'

        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug1.png'
        
        ax = plt.subplot(num_rows, num_cols, plot_idx)
        plt.title(title)
        
        img = load_image(path, 'RGB')
        plt.imshow(img)

        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)
        

def plot_spectrogram_aug(root: str, split_data: dict, split_type: str, label_data: dict, num_samples: int) -> None:
    '''
    Plots augmentations of a random spectrogram

    param: root = root folder name
    param: split_data = split samples
    param: split_type = type of split
    param: label_data = labels of samples
    param: num_samples = number of samples to plot

    return: None
    '''

    num_cols = 3
    num_rows = int(np.ceil(num_samples*3/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)
    
    plot_idx = 1
    idx = 1
    
    while plot_idx <= num_samples*3:
        file_name = np.random.choice(split_data[split_type])
        data = label_data[file_name]
        label = data[0]
        
        title = f'{idx}: {file_name} | Original'
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug1.png'
        plt.subplot(num_rows, num_cols, plot_idx)
        img = Image.open(path)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
        title = f'{idx}: {file_name} | Frequency Mask'
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug5.png'
        plt.subplot(num_rows, num_cols, plot_idx+1)
        img = Image.open(path)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])

        title = f'{idx}: {file_name} | Time Mask'
        path = f'{root}/{split_type.capitalize()}/{label}/{file_name}/{file_name}_aug6.png'             
        plt.subplot(num_rows, num_cols, plot_idx+2)
        img = Image.open(path)
        plt.imshow(img)
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
        
        idx += 1
        plot_idx += 3


def show_batch(root: str, name: str, label_data: dict, batch_data: tuple, color_map: str, img_size: tuple, num_samples: int) -> None:
    '''
    Show spectrograms of a random batch

    param: root = root folder name
    param: label_data = labels of samples
    param: batch_data = data related to the batches
    param: color_map = color map of image
    param: img_size = size of image in pixels
    param: num_samples = number of samples
    
    return: None
    '''

    split_type, shuffled_data, batches, batch_size = batch_data

    num_batches = int(np.ceil(len(shuffled_data)/batch_size))
    batch_idx = np.random.choice([i for i in range(num_batches)])
    batch = batches[batch_idx]
   
    indices = torch.randperm(len(batch)).tolist()
    
    if len(batch) < num_samples:
        num_samples = len(batch)

    indices = indices[:num_samples]

    num_cols = 8
    num_rows = int(np.ceil(num_samples/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.suptitle(f'Displaying {num_samples} out of {len(batch)} samples in batch #{batch_idx}')
    
    plot_idx = 1

    if split_type == 'valid':
        split_type = 'train'
    
    for idx in indices:
        sample = batch[idx]
        file_name = sample[:-5]
        emotion = label_data[file_name][0]
        gender = label_data[file_name][1]
        others = []
        others.append(label_data[file_name][2])

        if name == 'iemocap':    
            others.append(label_data[file_name][3])
            others.append(label_data[file_name][4])


        title = f'{emotion} | {gender} \n'
        for o in others:
            title += f'| {o} '

        path = f'{root}/{split_type.capitalize()}/{emotion}/{file_name}/{sample}.png'
    
        ax = plt.subplot(num_rows, num_cols, plot_idx)
        img = load_image(path, color_map)
        img = img.resize(img_size)
        
        if color_map == 'L':
            plt.imshow(img, cmap='gray')
        else :
            plt.imshow(img)
            
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(title, fontsize = 8)
        
        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)
        

def show_test_samples(root: str, test_data: dict, label_data: dict, plot_params: tuple, num_samples: int) -> None:
    '''
    Show spectrograms of a random batch

    param: root = root folder name
    param: label_data = labels of samples
    param: batch_data = data related to the batches
    param: color_map = color map of image
    param: img_size = size of image in pixels
    param: num_samples = number of samples
    
    return: None
    '''

    name, color_map, img_size, augment = plot_params
   
    indices = torch.randperm(len(test_data)).tolist()
    indices = indices[:num_samples]

    num_cols = 8
    num_rows = int(np.ceil(num_samples/num_cols))
    
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.suptitle(f'Displaying {num_samples} samples in test set')
    
    plot_idx = 1

    for idx in indices:
        sample = test_data[idx]
        file_name = sample
        if augment:
          file_name = sample[:-5]

        emotion = label_data[file_name][0]
        gender = label_data[file_name][1]
        others = []
        others.append(label_data[file_name][2])

        if name == 'iemocap':    
            others.append(label_data[file_name][3])
            others.append(label_data[file_name][4])


        title = f'{emotion} | {gender} \n'
        if augment:
          title = f'{emotion} ({sample[-1]}) | {gender} \n'
        for o in others:
            title += f'| {o} '

        path = f'{root}/Test/{emotion}/{file_name}/{file_name}_aug1.png'
        if augment:
          path = f'{root}/Test/{emotion}/{file_name}/{sample}.png'
    
        ax = plt.subplot(num_rows, num_cols, plot_idx)
        img = load_image(path, color_map)
        img = img.resize(img_size)
        
        if color_map == 'L':
            plt.imshow(img, cmap='gray')
        else :
            plt.imshow(img)
            
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        plt.title(title, fontsize = 8)
        
        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)
        
