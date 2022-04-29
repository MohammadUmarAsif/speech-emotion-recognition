import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import seaborn as sb
from IPython.display import display
from helper.visualization import hide_empty
from helper.common import load_data


COLORS = {'primary': '#003d7a', 'secondary': '#ff8400'}
HSPACE = 0.4
WSPACE = 0.25
ROTATION = 20
OVERALL_TITLE = {1: 'Training Loss', 2: 'Validation Loss'}
CATEGORY_TITLE = {1: 'Training Loss', 2: 'Validation Loss', 3: 'Accuracy', 4: 'F1 Score'}
NUMERICAL_TITLE = {1: 'Training Loss', 2: 'Validation Loss', 3: 'RMSE'}
FIG_WIDTH = 18
FIG_HEIGHT = 4
BAR_WIDTH = 0.3

def compare_simple(x: tuple, x_name: str, y: tuple, y_name: str, epochs: int) -> None:
    '''
    Plots comparison of 2 simple models

    param: x = plotting data of model x
    param: x_name = name of model x
    param: y = plotting data of model y
    param: y_name = name of model y
    param: epochs = number of epochs

    return: None
    '''

    data_params = (x, x_name, y, y_name, epochs)
    plot_type = 'simple'

    compare_durations(data_params)
    compare_curves(data_params, 'Emotion', 4, None, (0, 1), CATEGORY_TITLE, plot_type)


def compare_complex(x: tuple, x_name: str, y: tuple, y_name: str, epochs: int) -> None:
    '''
    Plots comparison of 2 complex models

    param: x = plotting data of model x
    param: x_name = name of model x
    param: y = plotting data of model y
    param: y_name = name of model y
    param: epochs = number of epochs

    return: None
    '''

    data_params = (x, x_name, y, y_name, epochs)
    plot_type = 'complex'

    compare_durations(data_params)
    compare_curves(data_params, 'Overall', 2, 0, None, OVERALL_TITLE, plot_type)
    compare_curves(data_params, 'Emotion', 4, 1, (0, 1), CATEGORY_TITLE, plot_type)
    compare_curves(data_params, 'Gender', 4, 2, (2, 3), CATEGORY_TITLE, plot_type)
    compare_curves(data_params, 'Valence', 3, 3, (4), NUMERICAL_TITLE, plot_type)
    compare_curves(data_params, 'Activation', 3, 4, (5), NUMERICAL_TITLE, plot_type)
    compare_curves(data_params, 'Dominance', 3, 5, (6), NUMERICAL_TITLE, plot_type)


def compare_hybrid(x: tuple, x_name: str, y: tuple, y_name: str, epochs: int) -> None:
    '''
    Plots comparison of a simple and a complex model

    param: x = plotting data of model x
    param: x_name = name of model x
    param: y = plotting data of model y
    param: y_name = name of model y
    param: epochs = number of epochs

    return: None
    '''
     
    data_params = (x, x_name, y, y_name, epochs)
    plot_type = 'hybrid'

    compare_durations(data_params)
    compare_curves(data_params, 'Emotion', 4, 1, (0, 1), CATEGORY_TITLE, plot_type)


def plot_complex(x: tuple, x_name: str, epochs: int) -> None:
    '''
    Plots data of a complex model

    param: x = plotting data of model x
    param: x_name = name of model x
    param: epochs = number of epochs

    return: None
    '''

    data_params = (x, x_name, epochs)

    display_duration(data_params)
    plot_curve(data_params, 'Overall', 2, 0, None, OVERALL_TITLE)
    plot_curve(data_params, 'Emotion', 4, 1, (0, 1), CATEGORY_TITLE)
    plot_curve(data_params, 'Gender', 4, 2, (2, 3), CATEGORY_TITLE)
    plot_curve(data_params, 'Valence', 3, 3, (4), NUMERICAL_TITLE)
    plot_curve(data_params, 'Activation', 3, 4, (5), NUMERICAL_TITLE)
    plot_curve(data_params, 'Dominance', 3, 5, (6), NUMERICAL_TITLE)


def compare_durations(data_params: tuple) -> None:
    '''
    Compares duration related information of 2 models

    param: data_params = data of models

    return: None
    '''

    x, x_name, y, y_name, epochs = data_params
    x_sum = sum(x[3])
    y_sum = sum(y[3])
    x_avg = x_sum/len(x[3])
    y_avg = y_sum/len(y[3])
    
    print(f'Total Epochs = {epochs}')
    print(f'Model {x_name} | Training Duration = {x_sum:.2f} sec | Average Duration = {x_avg:.2f} sec')
    print(f'Model {y_name} | Training Duration = {y_sum:.2f} sec | Average Duration = {y_avg:.2f} sec')


def display_duration(data_params: tuple) -> None:
    '''
    Displays duration related information of a model

    param: data_params = data of models

    return: None
    '''

    x, x_name, epochs = data_params
    x_sum = sum(x[3])
    x_avg = x_sum/len(x[3])
    
    if type(epochs) == int:
        print(f'Total Epochs = {epochs}')
    elif type(epochs) == tuple:
        print(f'Epoch Range = {epochs[0]} to {epochs[1]}')

    print(f'Model {x_name} | Training Duration = {x_sum:.2f} sec | Average Duration = {x_avg:.2f} sec')


def compare_curves(data_params: tuple, title: str, num_plots: int, loss_idx: int, metric_idx: tuple, 
                    title_dict: dict, plot_type: str) -> None:
    '''
    Plots curves for comparison of 2 models

    param: data_params = data of models
    param: title = super title
    param: num_plots = number of plots
    param: loss_idx = index of loss
    param: metric_idx = index of metrics
    param: title_dict = plot title
    param: plot_type = type of plot
    param: x_jump = step_size for x_ticks

    return: None
    '''
    
    _, x_name, _, y_name, epochs = data_params
    
    x_data, y_data = extract_data_two(data_params, num_plots, loss_idx, metric_idx, plot_type)

    tick = np.arange(0, epochs)
    tick_labels = np.arange(1, epochs+1)

    num_rows = 1
    num_cols = num_plots
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)
    plt.suptitle(title)
        
    plot_idx = 1

    for i in range(num_plots):
        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(x_data[i], color=COLORS['primary'])
        plt.plot(y_data[i], color=COLORS['secondary'])
            
        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.legend(title = 'Model', labels = [x_name, y_name])
        plt.title(title_dict[i+1])
            
        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)


def extract_data_two(data_params: tuple, num_plots: int, loss_idx: int, metric_idx: tuple, plot_type: str) -> tuple:
    '''
    Extracts the relevant data for 2 models

    param: data_params = data of models
    param: num_plots = number of plots
    param: loss_idx = index of loss
    param: metric_idx = index of metrics
    param: plot_type = type of plot

    return: relevant data of 2 models
    '''
    
    x, _, y, _, epochs = data_params
    x_data = []
    y_data = []
    
    for _ in range(num_plots):
      x_data.append([])
      y_data.append([])

    if plot_type == 'simple' or plot_type == 'complex':
        if loss_idx != None:
            for e in range(epochs):
                for i in range(2):
                    x_data[i].append(x[i][e][loss_idx])
                    y_data[i].append(y[i][e][loss_idx])
        else:
            for e in range(epochs):
                for i in range(2):
                    x_data[i].append(x[i][e])
                    y_data[i].append(y[i][e])
        
        if metric_idx != None:
            if type(metric_idx) == int:
                metric_idx = [metric_idx]
            for e in range(epochs):
                for idx, m_idx in enumerate(metric_idx):
                    x_data[idx+2].append(x[2][e][m_idx])
                    y_data[idx+2].append(y[2][e][m_idx])
    
    elif plot_type == 'hybrid':
        for e in range(epochs):
            for i in range(2):
                x_data[i].append(x[i][e])
                y_data[i].append(y[i][e][loss_idx])

        for e in range(epochs):
            for idx, m_idx in enumerate(metric_idx):
                x_data[idx+2].append(x[2][e][m_idx])
                y_data[idx+2].append(y[2][e][m_idx])

    return (x_data, y_data)


def extract_data_one(data_params: tuple, num_plots: int, loss_idx: int, metric_idx: tuple) -> tuple:
    '''
    Extracts the relevant data for 1 model

    param: data_params = data of model
    param: num_plots = number of plots
    param: loss_idx = index of loss
    param: metric_idx = index of metrics
    param: checkpoint_num = number of training checkpoint
    param: epoch_size = number of epochs in each session

    return: relevant data of the model
    '''
    
    x, _, epochs = data_params
    x_data = []

    for _ in range(num_plots):
      x_data.append([])

    if loss_idx != None:
        for e in range(epochs):
            for i in range(2):
                x_data[i].append(x[i][e][loss_idx])

    if metric_idx != None:
        for e in range(epochs):
            for idx, m_idx in enumerate(metric_idx):
                x_data[idx+2].append(x[2][e][m_idx])
                    
    return x_data


def plot_curve(data_params: tuple, title: str, num_plots: int, loss_idx: int, metric_idx: tuple, title_dict: dict, x_jump = 1) -> None:
    '''
    Plots curves for a model

    param: data_params = data of models
    param: title = super title
    param: num_plots = number of plots
    param: loss_idx = index of loss
    param: metric_idx = index of metrics
    param: title_dict = plot title
    param: x_jump = step_size for x_ticks
    param: best_epoch = epoch of best model
    param: checkpoint_num = number of training checkpoint
    param: epoch_size =  number of epochs in each session

    return: None
    '''
    
    _, x_name, epochs = data_params

    tick = np.arange(0, epochs, x_jump)
    tick_labels = np.arange(1, epochs+1, x_jump)
    
    x_data = extract_data_one(data_params, num_plots, loss_idx, metric_idx)

    num_rows = 1
    num_cols = num_plots
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)
    plt.suptitle(title)
        
    plot_idx = 1

    for i in range(num_plots):
        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(x_data[i], color = COLORS['primary'])
        plt.legend(title = 'Model', labels = [x_name])
        
        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.title(title_dict[i+1])
            
        plot_idx += 1

    hide_empty(plot_idx, num_rows, num_cols)


def extract_train_data(x: tuple,  epochs, num_data:int, loss_idx = None, metric_idx = None) -> tuple:
    '''
    Extracts the relevant data for 1 model

    param: data_params = data of model
    param: num_plots = number of plots
    param: loss_idx = index of loss
    param: metric_idx = index of metrics
    param: checkpoint_num = number of training checkpoint
    param: epoch_size = number of epochs in each session

    return: relevant data of the model
    '''
    
    x_data = []

    for _ in range(num_data):
      x_data.append([])

    if loss_idx != None:
        for e in range(epochs):
            for i in range(num_data):
                x_data[i].append(x[i][e][loss_idx])

    if metric_idx != None:
        if type(metric_idx) == int:
            metric_idx = [metric_idx]
        for e in range(epochs):
            for i, m_idx in enumerate(metric_idx):
                x_data[i].append(x[e][m_idx])
                    
    return x_data


def plot_train_loss(checkpoint: dict, x_name: str, best_epoch: int, checkpoint_num: int, epoch_size: int, x_jump = 1) -> None:
    '''
    Plots the training and validation loss for each category

    param: checkpoint = plotting data
    param: x_name = session name
    param: best_epoch = best epoch number
    param: checkpoint_num = session number
    param: epoch_size = number of epochs in each session
    param: x_jump = jump on x-tick

    return: None
    '''

    x = checkpoint['plotting data']
    epochs = checkpoint['current epoch']

    x_sum = sum(x[2])
    x_avg = x_sum/len(x[2])
    
    print(f'Epoch Range = {epochs[0]} to {epochs[1]}')
    print(f'{x_name} | Training Duration = {x_sum:.2f} sec | Average Duration = {x_avg:.2f} sec')
    

    titles = ['Overall', 'Emotion', 'Gender', 'Valence', 'Activation', 'Dominance']
    
    tick = np.arange(0, epochs[1] - epoch_size*(checkpoint_num - 1), x_jump)
    tick_labels = np.arange(epochs[0]+1, epochs[1]+1, x_jump)
    
    epochs = epochs[1] - epoch_size*(checkpoint_num - 1)
    
    plot_idx = 1
    num_cols = 2
    num_rows = int(np.ceil(len(titles)/num_cols))
    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)


    for idx, title in enumerate(titles):
        x_data = extract_train_data(x, epochs, num_data = 2, loss_idx = idx)

        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(x_data[0], color = COLORS['primary'])
        plt.plot(x_data[1], color = COLORS['secondary'])

        if best_epoch != None:
            plt.scatter(best_epoch, x_data[0][best_epoch], color = 'black')
            plt.scatter(best_epoch, x_data[1][best_epoch], color = 'black')
            plt.legend(title = 'Model', labels = ['Training Loss', 'Validation Loss', 'Best'])
        else:
            plt.legend(title = 'Model', labels = ['Training Loss', 'Validation Loss'])

        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.title(f'{title} Loss')

        plot_idx += 1
    
    hide_empty(plot_idx, num_rows, num_cols)


def plot_train_classification(checkpoint: dict, x_name: str, best_epoch: int, checkpoint_num: int,
                    epoch_size: int, average: str, x_jump = 1) -> None:
    '''
    Plots the metrics for classification tasks

    param: checkpoint = plotting data
    param: x_name = session name
    param: best_epoch = best epoch number
    param: checkpoint_num = session number
    param: epoch_size = number of epochs in each session
    param: average = averaging type
    param: x_jump = jump on x-tick

    return: None
    '''

    titles = [f'Emotion ({average.capitalize()})', f'Gender ({average.capitalize()})']
    average += ' metric'
    x = checkpoint[average]
    epochs = checkpoint['current epoch']

    tick = np.arange(0, epochs[1] - epoch_size*(checkpoint_num - 1), x_jump)
    tick_labels = np.arange(epochs[0]+1, epochs[1]+1, x_jump)
    epochs = epochs[1] - epoch_size*(checkpoint_num - 1)

    for idx, title in enumerate(titles):
        plot_idx = 1
        num_rows = 1
        num_cols = 2
        plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
        plt.subplots_adjust(hspace = HSPACE)
        plt.suptitle(title)

        metric_idx = None

        if idx == 0:
            metric_idx = (0, 1)
        elif idx == 1:
            metric_idx = (2, 3)

        x_data = extract_train_data(x, epochs, num_data = 2, metric_idx = metric_idx)

        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(x_data[0], color = COLORS['primary'])
        if best_epoch != None:
            plt.scatter(best_epoch, x_data[0][best_epoch], color = COLORS['secondary'])
            plt.legend(title = 'Model', labels = [x_name, 'Best'])
        else:
            plt.legend(title = 'Model', labels = [x_name])

        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.title('Accuracy')

        plt.subplot(num_rows, num_cols, plot_idx + 1)
        plt.plot(x_data[1], color = COLORS['primary'])
        if best_epoch != None:
            plt.scatter(best_epoch, x_data[1][best_epoch], color = COLORS['secondary'])
            plt.legend(title = 'Model', labels = [x_name, 'Best'])
        else:
            plt.legend(title = 'Model', labels = [x_name])

        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.title('F1 Score')
            
        plot_idx += 2


def plot_train_regression(checkpoint: dict, x_name: str, best_epoch: int, checkpoint_num: int,
                    epoch_size: int, x_jump = 1) -> None:
    '''
    Plots the curves for regression tasks

    param: checkpoint = plotting data
    param: x_name = session name
    param: best_epoch = best epoch number
    param: checkpoint_num = session number
    param: epoch_size = number of epochs in each session
    param: x_jump = jump on x-tick

    return: None
    '''
    
    x = checkpoint['numerical metric']
    epochs = checkpoint['current epoch']
    titles = ['Valence - RMSE', 'Arousal - RMSE', 'Dominance - RMSE']
    
    tick = np.arange(0, epochs[1] - epoch_size*(checkpoint_num - 1), x_jump)
    tick_labels = np.arange(epochs[0]+1, epochs[1]+1, x_jump)
    epochs = epochs[1] - epoch_size*(checkpoint_num - 1)

    plot_idx = 1

    num_rows = 2
    num_cols = 2

    plt.subplots(num_rows, num_cols, figsize = (FIG_WIDTH, num_rows*FIG_HEIGHT))
    plt.subplots_adjust(hspace = HSPACE)
    

    for idx, title in enumerate(titles):
        x_data = extract_train_data(x, epochs, num_data = 1, metric_idx = idx)

        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(x_data[0], color = COLORS['primary'])
        if best_epoch != None:
            plt.scatter(best_epoch, x_data[0][best_epoch], color = COLORS['secondary'])
            plt.legend(title = 'Model', labels = [x_name, 'Best'])
        else:
            plt.legend(title = 'Model', labels = [x_name])
        plt.xlabel('Epoch')    
        plt.xticks(tick, tick_labels)
        plt.title(title)
            
        plot_idx += 1
    
    hide_empty(plot_idx, num_rows, num_cols)


def plot_confusion_matrix(conf_matrix: torch.tensor, labels: tuple, title: str, fig_size: tuple, wspace: float) -> None:
    '''
    Plots a confusion matrix

    param: conf_matrix = confusion matrix
    param: labels = labels of matrix
    param: title = plot title
    param: fig_size = size of figure
    param: wspace = horizontal space between plots
    
    return: None
    '''
    
    conf_matrix = conf_matrix.to('cpu')
    percentage = conf_matrix/torch.sum(conf_matrix)

    plt.subplots(1, 2, figsize = fig_size)
    plt.subplots_adjust(wspace = wspace)

    plt.subplot(1, 2, 1)
    ax = sb.heatmap(conf_matrix, cbar = False, cmap = 'Blues', annot = True, fmt = 'd', yticklabels = labels)
    ax.set(title=f'{title} (Amount)', xlabel = 'Predicted Value', ylabel = 'Actual Value')
    ax.set_xticklabels(labels = labels, rotation = ROTATION)

    plt.subplot(1, 2, 2)
    ax = sb.heatmap(conf_matrix, cbar = False, cmap = 'Blues', annot = percentage, fmt = '.1%', yticklabels = labels)
    ax.set(title=f'{title} (Percentage)', xlabel = 'Predicted Value', ylabel = 'Actual Value')
    ax.set_xticklabels(labels = labels, rotation = ROTATION)


def extract_class_data(x: list, num_labels: int, metric_range: tuple) -> tuple:   
    '''
    Extracts the relevant data of classes

    param: x = data of model
    param: num_labels = number of labels 
    param: metric_range = indices of metric

    return: relevant data of the model
    '''
    
    acc_data = []
    f1_data = []

    for _ in range(num_labels):
      acc_data.append([])
      f1_data.append([])

    for k in range(0, num_labels):
        acc_data[k] = x[metric_range[0]][k].to('cpu')
        f1_data[k] = x[metric_range[1]][k].to('cpu')

    return (acc_data, f1_data)


def plot_per_class(data: dict, class_labels: tuple, class_type: str) -> None:
    '''
    Plots the best model metrics for each class 

    param: data = data of best model
    param: class_labels = labels of classes
    param: class_type = type of class

    return: None 
    '''

    metric_range = None
    width = FIG_WIDTH

    if class_type == 'Emotion':
        metric_range = (0, 1)
    elif class_type == 'Gender':
        metric_range = (2, 3) 
        width = int(width/2)
    elif class_type == 'Intensity':
        metric_range = (4, 5) 
        width = int(width/2)

    extracted_data = extract_class_data(data['class metric'], len(class_labels), metric_range)

    num_cols = 2
    num_rows = 1
    plt.subplots(num_rows, num_cols, figsize = (width, num_rows*FIG_HEIGHT))
    
    for idx, data in enumerate(extracted_data):
        plt.subplot(num_rows, num_cols, idx + 1)
        plots = plt.bar(class_labels, data, color = COLORS['primary'])
        plt.xlabel('Labels')    
        
        if idx == 0:
            plt.title('Accuracy')
        elif idx == 1:
            plt.title('F1 Score')

        plt.xticks(rotation = ROTATION)

        max_bar = max([bar.get_height() for bar in plots.patches])
        for idx, bar in enumerate(plots.patches):    
            plt.annotate(f'{bar.get_height()*100:.1f}', (bar.get_x()+bar.get_width()/2, 
                            bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
        plt.ylim(0, max_bar + 0.1*max_bar)


def plot_maps(params: tuple, plot_type: str, fig_size: tuple) -> None:
    '''
    Plots the maps (feature maps or attention weights)

    param: params = parameters for plotting
    param: plot_type = type of plot
    param: fig_size = size of figure

    return: None
    '''
    
    filters = None
    num_filters = None
    sup_title, map_title = None, None
    
    if plot_type == 'filters':
      model, conv_idx = params
      filters = model.conv_list[conv_idx - 1].weight
      num_filters = filters.shape[0]
      sup_title = f'{num_filters} filters in Convolutional Layer {conv_idx}'
      map_title = 'Filter'

    elif plot_type == 'attention' or plot_type == 'feature':
      weights, idx = params
      filters = weights[idx].squeeze()
      num_filters = filters.shape[0]
    
      if plot_type == 'attention':
        sup_title = f'{num_filters} attention weights'
        map_title = 'Weight'
      elif plot_type == 'feature':
        sup_title = f'{num_filters} feature maps in Convolutional Layer {idx + 1}'
        map_title = 'Map'
    
    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    num_filters = filters.shape[0]

    num_cols = 12
    num_rows = int(np.ceil(num_filters/num_cols))
    plt.subplots(num_rows, num_cols, figsize = fig_size)

    plt.suptitle(sup_title)

    plot_idx = 1
    
    for i in range(num_filters):
       
        filter = filters[i].detach()
        ax = plt.subplot(num_rows, num_cols, plot_idx)
        ax.set_xticks([])
        ax.set_yticks([])
        
        
        if plot_type == 'filters':
          plt.imshow(filter[0], cmap = 'gray')
        else:
          plt.imshow(filter, cmap = 'gray')
        plt.title(f'{map_title} #{i+1}')
        plot_idx += 1
    
    hide_empty(plot_idx, num_rows, num_cols)


def display_test_data(data: list, task: list, columns: list, task_type: str) -> None:
    '''
    Displays the metrics in a tabular format

    param: data = data to display
    param: task = tasks (emotion, gender, intensity)
    param: columns = columns (metrics) 
    param: task_type = type of task
    
    return: None
    '''

    data_dic = {}
    for i, col in enumerate(columns):
      data_dic[col] = []
      if i == 0:
        data_dic[col].extend(task)   

    index = None

    if task_type == 'classification':
      for i, value in enumerate(data):
        if i in [0, 1]:
            if i == 1:
                data_dic[columns[i+1]].append(f'{value:.4f}')
            else:
                data_dic[columns[i+1]].append(f'{value*100:.2f}')
        elif i in [2, 3]:
            if i == 3:
                data_dic[columns[i-1]].append(f'{value:.4f}')
            else:
                data_dic[columns[i-1]].append(f'{value*100:.2f}')
        elif i in [4, 5]:
            if i == 5:
                data_dic[columns[i-3]].append(f'{value:.4f}')
            else:
                data_dic[columns[i-3]].append(f'{value*100:.2f}')

      index = [i+1 for i in range(int(len(data)/2))]
    
    elif task_type == 'regression':
      for i, value in enumerate(data):
        data_dic[columns[1]].append(f'{value:.4f}')
      index = [i+1 for i in range(len(data))]
    
    display(pd.DataFrame(data_dic, columns=columns, index = index))


def plot_class_pred(label: list, values: list, title: str, actual: str, top_k: int) -> None:
    '''
    Plots the classification predictions

    param: label = class names
    param: values = prediction values
    param: title = type of task
    param: actual = actual label
    param: top_k = number of top values to consider

    return: None
    '''

    plots = plt.bar(label, values, width = BAR_WIDTH + 0.2, color = COLORS['secondary']);
    plt.ylabel('Probability')
    plt.xlabel(title)
    plt.title(f'{title} Predictions - ({actual})')
    
    height_list = sorted([bar.get_height() for bar in plots.patches])
    max_bar = height_list[-1]
    pred_list = height_list[-top_k:]
    
    for bar in plots.patches: 
      if bar.get_height() in pred_list:
        bar.set_color(COLORS['primary'])
      plt.annotate(f'{bar.get_height():.2f}', (bar.get_x()+bar.get_width()/2, 
                          bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
    plt.ylim(0, max_bar + 0.1*max_bar)

    if title != 'Emotion':
      plt.legend(handles = [plots, plots], labels = ['Predicted', 'Other'])
      ax = plt.gca()
      leg = ax.get_legend()
      leg.legendHandles[0].set_color(COLORS['primary'])
      leg.legendHandles[1].set_color(COLORS['secondary'])
    
    else:
      plt.legend(handles = [plots, plots], labels = [f'Top {top_k}', 'Others'])
      ax = plt.gca()
      leg = ax.get_legend()
      leg.legendHandles[0].set_color(COLORS['primary']) 
      leg.legendHandles[1].set_color(COLORS['secondary'])


def plot_num_pred(labels: list, preds: list, pred_labels: list) -> None:
    '''
    Plots the numerical predictions

    param: labels = class names
    param: preds = predicted values
    param: pred_labels = actual values

    return: None
    '''

    x = np.arange(len(labels)) 

    plots = plt.bar(x - BAR_WIDTH/2, pred_labels[2:], BAR_WIDTH, label = 'Actual', color = COLORS['primary'])
    
    max_bar1 = max([bar.get_height() for bar in plots.patches])
    for bar in plots.patches: 
        plt.annotate(f'{bar.get_height():.1f}', (bar.get_x()+bar.get_width()/2, 
                            bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')

    plots = plt.bar(x + BAR_WIDTH/2, preds[2:], BAR_WIDTH, label = 'Predicted', color = COLORS['secondary'])
    max_bar2 = max([bar.get_height() for bar in plots.patches])
    for bar in plots.patches: 
        plt.annotate(f'{bar.get_height():.2f}', (bar.get_x()+bar.get_width()/2, 
                            bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
    
    max_bar = max(max_bar2, max_bar1)
    plt.ylim(0, max_bar + 0.1*max_bar)

    plt.ylabel('Value')
    plt.title('Dimensional Predictions')
    plt.xticks(x, labels)
    plt.legend()


def plot_iemocap_prediction(roots: tuple, preds: list, pred_labels: list, top_k: int) -> None:
    '''
    Plots predictions for an iemocap sample

    param: roots = root folder names
    param: preds = predictions
    param: pred_labels = labels of predictions
    param: top_k = number of top predictions to consider
    
    return: None
    '''
  
    _, meta_root = roots
    class_data = load_data(meta_root, 'Class')

    for i in range(0, 2):
        preds[i] = torch.exp(preds[i])
    for i in range(2, 5):
        pred_labels[i] = pred_labels[i][0][0].item()
        preds[i] = preds[i][0][0].item()

    fig = plt.figure(figsize=(18, 4)) 
    plt.subplots_adjust(wspace = WSPACE)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 2, 3])
    
    plt.subplot(gs[0])
    class_labels = list(class_data.keys())

    plot_class_pred(class_labels[:-2], preds[0][0].detach(), 'Emotion', class_labels[:-2][pred_labels[0][0]], top_k)
    
    plt.subplot(gs[1])
    plot_class_pred(class_labels[-2:], preds[1][0].detach(), 'Gender', class_labels[-2:][pred_labels[1][0]], top_k = 1)

    plt.subplot(gs[2])
    plot_num_pred(['Valence', 'Arousal', 'Dominance'], preds, pred_labels)


def plot_ravdess_prediction(roots: tuple, preds: list, pred_labels: list, top_k: int) -> None:
    '''
    Plots predictions for a ravdess sample

    param: roots = root folder names
    param: preds = predictions
    param: pred_labels = labels of predictions
    param: top_k = number of top predictions to consider
    
    return: None
    '''

    _, meta_root, _ = roots
    class_data = load_data(meta_root, 'Class')

    for i in range(0, 3):
        preds[i] = torch.exp(preds[i])

    fig = plt.figure(figsize=(18, 4)) 
    plt.subplots_adjust(wspace = WSPACE)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 2, 1])
    
    plt.subplot(gs[0])
    class_labels = list(class_data.keys())

    plot_class_pred(class_labels[:-4], preds[0][0].detach(), 'Emotion', class_labels[:-4][pred_labels[0][0]], top_k)
    
    plt.subplot(gs[1])
    plot_class_pred(class_labels[-4:-2], preds[1][0].detach(), 'Gender', class_labels[-4:-2][pred_labels[1][0]], top_k = 1)

    plt.subplot(gs[2])
    plot_intensity_pred(3.5, preds[3][0].detach(), 'Intensity', class_labels[-2:][pred_labels[2][0]])


def plot_intensity_pred(threshold: int, pred_value: float, title: str, actual: str) -> None:
    '''
    Plots prediction for intensity
    
    param: threshold = threshold for intensity value
    param: pred_value = predicted value
    param: title = title of plot
    param: actual = actual value

    return: None
    '''
    
    plots = plt.bar(['Prediction'], pred_value, color = COLORS['secondary']);
    plt.ylabel('Value')
    plt.xlabel(title)
    plt.title(f'{title} Predictions - ({actual})')
    
    height_list = sorted([bar.get_height() for bar in plots.patches])
    max_bar = height_list[-1]
    
    line = plt.axhline(y = threshold, color = COLORS['primary'])

    for bar in plots.patches: 
      plt.annotate(f'{bar.get_height():.2f}', (bar.get_x()+bar.get_width()/2, 
                          bar.get_height()), ha = 'center', va = 'center', size = 10, xytext = (0,8), textcoords = 'offset points')
    
    max_bar = max(max_bar, 6)
    plt.ylim(0, max_bar + 0.1*max_bar)

    plt.legend(handles = [plots], labels = ['Strong'] if pred_value >= threshold else ['Normal'])


def plot_img_pred(file_name: str, img: torch.Tensor, color_map: str, aug_type: str) -> None:
    '''
    Plots an image

    param: file_name = name of file 
    param: img = img data
    param: color_map = color map
    param: aug_type = augmentation type

    return: None
    '''

    img = img[0].squeeze().detach()
    
    if color_map == 'L':
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    
    plt.title(f'{file_name}_aug{aug_type}')
    plt.xticks([])
    plt.yticks([])


def plot_classwise_pred(data: list, labels: list, title: str, fig_size: tuple) -> None:
    '''
    Plots classwise predictions of spectrograms for a task

    param: data = prediction data
    param: labels = labels of classes
    param: title = type of task
    param: fig_size = size of figure
    
    return: None
    '''

    num_cols = 2
    num_rows = int(np.ceil(len(labels)/num_cols))
    plt.subplots(num_rows, num_cols, figsize = fig_size)
    plt.subplots_adjust(hspace = 0.4)
    plt.suptitle(f'Spectrogram Predictions for Respective {title}')
    plot_idx = 1
    
    for key, value in data.items():
        if len(value) == 0:
            continue
        plt.subplot(num_rows, num_cols, plot_idx)
        plt.plot(value, color = COLORS['primary'])
        plt.title(labels[key])
        plot_idx += 1
        plt.ylim(0, 1)
        plt.ylabel('Probability')
        plt.xlabel('Spectrogram')
        tick = [i for i in range(0, len(value), int(len(value)/5) if int(len(value)/5) != 0 else 1)]
        tick_labels = [i for i in range(1, len(value)+1, int(len(value)/5) if int(len(value)/5) != 0 else 1)]
        plt.xticks(tick, tick_labels)
    
    hide_empty(plot_idx, num_rows, num_cols)

