import time
import torch
import torch.optim as optim
import numpy as np
from ipywidgets import IntProgress, Layout
from IPython.display import display
from helper.network import inverse_scaler, ensemble_preds
from helper.common import load_data
from batch import get_labels, get_current_batch, load_iemocap_batch, get_augmentations, load_ravdess_batch
from model import SimpleNetwork, ComplexNetwork


def get_simple_model(model_info: dict, transforms: list, num_classes: int) -> tuple:
    '''
    Gets the model parameters for training

    param: model_info = information for creating model
    param: transforms = normalization transforms
    param: num_classes = number of emotion classes
    
    return: model, optimizer, and transforms
    '''

    all_transforms = [model_info['input']]
    all_transforms.extend(transforms)

    model_info['convolutional'][0]['input'] = 1 if transforms[-1] == 'L' else 3
    model_info['output'] = num_classes
    model = SimpleNetwork(model_info)
    
    optimizer = optim.SGD(model.parameters(), lr = 0.01)
    
    return (model, optimizer, all_transforms)


def train_simple_network(roots: tuple, model_params: tuple, torch_params: dict, other_params: dict) -> tuple:
    '''
    Trains a simple network

    param: roots = names of root folders
    param: model_params = data related to model
    param: torch_params = data related to pytorch
    param: other_params = data related to other details

    return: losses, performance metrics, and training duration
    '''

    # Unpacking data
    root, meta_root = roots
    model, optimizer, transforms = model_params
    criterion, metrics = torch_params.values()
    epochs, batch_size = other_params.values()

    label_data = load_data(meta_root, 'Label')
    class_data = load_data(meta_root, 'Class')
    batch_data = load_data(meta_root, 'Batch Data (Dev)')
    
    train_batches, valid_batches = batch_data['data batches']
    shuffled_train, shuffled_valid = batch_data['shuffled data']
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    for metric in metrics:
        metric.to(device)

    batch_count = [0, 0]

    training_loss = []
    validation_loss = []
    performance_metric = []
    epoch_duration = []

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = epochs, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Epoch - {progress_bar.value} / {epochs}'
    display(progress_bar)

    split_type = 'train'
    
    for e in range(epochs):
        start = time.time()
        model.train()
        
        epoch_train_loss = 0
        epoch_valid_loss = 0
        epoch_metrics = [0]*len(metrics)

        for idx in range(len(train_batches)):
            batch = get_current_batch(train_batches, batch_size, batch_count[0], shuffled_train)
            images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 0, transforms)
            images, labels = images.to(device), labels.to(device)

            pred_logps = model.forward(images)

            pred_labels = get_labels(labels, 0, 'classification').to(device)
            overall_loss = criterion(pred_logps, pred_labels)

            epoch_train_loss += overall_loss.item()
            
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
    
        else:
            model.eval()

            with torch.no_grad():
                 for idx in range(len(valid_batches)):
                    batch = get_current_batch(valid_batches, batch_size, batch_count[1], shuffled_valid)
                    images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 1, transforms)
                    images, labels = images.to(device), labels.to(device)

                    pred_logps = model.forward(images)

                    pred_labels = get_labels(labels, 0, 'classification').to(device)
                    overall_loss = criterion(pred_logps, pred_labels)

                    epoch_valid_loss += overall_loss.item()

                    for idx, metric in enumerate(metrics):
                        epoch_metrics[idx] += metric(pred_logps, pred_labels).item()

        end = time.time()
        epoch_duration.append(end - start)

        training_loss.append(epoch_train_loss/len(train_batches))
        validation_loss.append(epoch_valid_loss/len(valid_batches))
        performance_metric.append([metric/len(valid_batches) for metric in epoch_metrics])
        
        batch_count = [0, 0]

        progress_bar.value = e + 1
        if progress_bar.value == epochs:
            progress_bar.description = f'Training Completed'
        else:
            progress_bar.description = f'Epoch - {progress_bar.value} / {epochs}'

    return (training_loss, validation_loss, performance_metric, epoch_duration)


def get_complex_model(model_info: dict, layer_params: dict, transforms: list, multitask_params: dict, optim_params: dict, task_type: tuple) -> tuple:
    '''
    Gets the model parameters for training

    param: model_info = information for creating model
    param: layer_params = parameters for model layers
    param: transforms = normalization transforms
    param: multitask_params = parameters for multitask learning
    param: optim_params = parameters for optimizer
    param: task_type = type of each task
    
    return: model, optimizer, and transforms
    '''

    all_transforms = [model_info['input']]
    all_transforms.extend(transforms)
    
    model = ComplexNetwork(model_info, layer_params, multitask_params, task_type)

    optimizer = None
    if optim_params['type'] == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr = optim_params['learning rate'], momentum = optim_params['momentum'], 
                                weight_decay = optim_params['weight decay'])
    elif optim_params['type'] == 'rmsprop':
        optimizer = optim.RMSprop(model.parameters(), lr = optim_params['learning rate'], momentum = optim_params['momentum'], 
                                weight_decay = optim_params['weight decay'])
    
    return (model, optimizer, all_transforms, optim_params['scheduler'])


def train_complex_network(roots: tuple, model_params: tuple, torch_params: dict, other_params: dict) -> tuple:
    '''
    Trains a complex network

    param: roots = names of root folders
    param: model_params = data related to model
    param: torch_params = data related to pytorch
    param: other_params = data related to other details

    return: losses, performance metrics, and training duration
    '''

    # Unpacking data
    root, meta_root = roots
    model, optimizer, transforms, scheduler_param = model_params
    criterion, metrics = torch_params.values()
    scaling_type, task_type, epochs, batch_size = other_params.values()

    label_data = load_data(meta_root, 'Label')
    class_data = load_data(meta_root, 'Class')
    batch_data = load_data(meta_root, 'Batch Data (Dev)')
    stats = load_data(meta_root, 'Statistics')
    scaled_vad = load_data(meta_root, 'Scaled VAD')
    
    train_batches, valid_batches = batch_data['data batches']
    shuffled_train, shuffled_valid = batch_data['shuffled data']

    scheduler = None
    if scheduler_param:
        milestones, gamma = scheduler_param
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones = milestones, gamma = gamma)
      
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    for metric in metrics:
        metric.to(device)

    batch_count = [0, 0]

    training_loss = []
    validation_loss = []
    performance_metric = []
    epoch_duration = []

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = epochs, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Epoch - {progress_bar.value} / {epochs}'
    display(progress_bar)

    split_type = 'train'
    
    for e in range(epochs):
        start = time.time()
        model.train()
        
        epoch_train_loss = [0]*(len(task_type) + 1)
        epoch_valid_loss = [0]*(len(task_type) + 1)
        epoch_metrics = [0]*len(metrics)

        for idx in range(len(train_batches)):
            batch = get_current_batch(train_batches, batch_size, batch_count[0], shuffled_train)
            images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 0, transforms, 
                                                    scaled_vad, scaling_type)
            images, labels = images.to(device), labels.to(device)

            preds = model.forward(images)
            pred_labels = []
            
            for idx, l_type in enumerate(task_type):
                pred_labels.append(get_labels(labels, idx, l_type).to(device))
            
            losses = []
            for idx, crit in enumerate(criterion):
                losses.append(crit(preds[idx], pred_labels[idx]))
            
            overall_loss = model.compute_loss(losses)

            epoch_train_loss[0] += overall_loss.item()
            for i in range(len(losses)):
                epoch_train_loss[i+1] += losses[i].item()
            
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
    
        else:
            model.eval()

            with torch.no_grad():
                 for idx in range(len(valid_batches)):
                    batch = get_current_batch(valid_batches, batch_size, batch_count[1], shuffled_valid)
                    images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 1, transforms, 
                                                            scaled_vad, scaling_type)
                    images, labels = images.to(device), labels.to(device)

                    preds = model.forward(images)
                    pred_labels = []
                    
                    for idx, l_type in enumerate(task_type):
                        pred_labels.append(get_labels(labels, idx, l_type).to(device))
                    
                    losses = []
                    for idx, crit in enumerate(criterion):
                        losses.append(crit(preds[idx], pred_labels[idx]))
                    
                    overall_loss = model.compute_loss(losses)
                    
                    epoch_valid_loss[0] += overall_loss.item()
                    for i in range(len(losses)):
                        epoch_valid_loss[i+1] += losses[i].item()
                    
                    inverse_scaler(preds, pred_labels, stats, scaling_type)

                    for idx, metric in enumerate(metrics):
                        if idx in [0, 1]:
                            epoch_metrics[idx] += metric(preds[0], pred_labels[0]).item()
                        elif idx in [2, 3]:
                            epoch_metrics[idx] += metric(preds[1], pred_labels[1]).item()
                        else:
                            epoch_metrics[idx] += metric(preds[idx-2], pred_labels[idx-2]).item()

        if scheduler_param:
            scheduler.step()
        
        end = time.time()
        epoch_duration.append(end - start)

        training_loss.append([loss/len(train_batches) for loss in epoch_train_loss])
        validation_loss.append([loss/len(valid_batches) for loss in epoch_valid_loss])
        performance_metric.append([metric/len(valid_batches) for metric in epoch_metrics])
        
        batch_count = [0, 0]

        progress_bar.value = e + 1
        if progress_bar.value == epochs:
            progress_bar.description = f'Training Completed'
        else:
            progress_bar.description = f'Epoch - {progress_bar.value} / {epochs}'

    return (training_loss, validation_loss, performance_metric, epoch_duration)


def train_network(roots: tuple, model_params: tuple, torch_params: dict, other_params: dict, checkpoint_num: int) -> None:
    '''
    Trains the final network

    param: roots = names of root folders
    param: model_params = data related to model
    param: torch_params = data related to pytorch
    param: other_params = data related to other details
    param: checkpoint_num = number of training checkpoint

    return: None
    '''

    # Unpacking data
    root, meta_root = roots
    model, optimizer, transforms, scheduler = model_params
    criterion, metrics, metrics_macro, metrics_none, metrics_num, confusion_matrix = torch_params.values()
    scaling_type, task_type, epochs, batch_size = other_params.values()
    
    label_data = load_data(meta_root, 'Label')
    class_data = load_data(meta_root, 'Class')
    scaled_vad = load_data(meta_root, 'Scaled VAD')
    stats = load_data(meta_root, 'Statistics')
    batch_data = load_data(meta_root, 'Batch Data (Train)')
    
    train_batches, valid_batches = batch_data['data batches']
    shuffled_train, shuffled_valid = batch_data['shuffled data']
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    for metric in metrics:
        metric.to(device)
    
    for metric in metrics_macro:
        metric.to(device)

    for metric in metrics_none:
        metric.to(device)
    
    for metric in metrics_num:
        metric.to(device)
    
    for matrix in confusion_matrix:
        matrix.to(device)

    batch_count = [0, 0]

    training_loss = []
    validation_loss = []
    performance_metric = []
    performance_metric_macro = []
    epoch_duration = []
    performance_metric_num = []

    # Displaying progress
    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = epochs[0], max = epochs[1], style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = f'Epoch - {progress_bar.value} / {epochs[1]}'
    display(progress_bar)

    split_type = 'train'

    min_validation_loss = float('inf')
    if checkpoint_num > 1:
        current_best =  torch.load(f'{meta_root}/Best.pt')
        min_validation_loss = current_best['plotting data'][1][0]
    
    for e in range(epochs[0], epochs[1]):
        start = time.time()
        model.train()
        
        epoch_train_loss = [0]*(len(task_type) + 1)
        epoch_valid_loss = [0]*(len(task_type) + 1)
        epoch_metrics = [0]*len(metrics)
        epoch_metrics_macro = [0]*len(metrics_macro)
        epoch_metrics_none = [0]*len(metrics_none)
        epoch_metrics_num = [0]*len(metrics_num)
        epoch_confusion = [0]*len(confusion_matrix)
        emo_div_amount_none = [len(valid_batches)]*(len(class_data)-2)
        gen_div_amount_none = [len(valid_batches)]*2

        for idx in range(len(train_batches)):
            batch = get_current_batch(train_batches, batch_size, batch_count[0], shuffled_train)
            images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 0, transforms, 
                                                    scaled_vad, scaling_type)
            images, labels = images.to(device), labels.to(device)

            preds, _, _ = model.forward(images)
            pred_labels = []
            
            for idx, l_type in enumerate(task_type):
                pred_labels.append(get_labels(labels, idx, l_type).to(device))
            
            losses = []
            for idx, crit in enumerate(criterion):
                losses.append(crit(preds[idx], pred_labels[idx]))
            
            overall_loss = model.compute_loss(losses)

            epoch_train_loss[0] += overall_loss.item()
            for i in range(len(losses)):
                epoch_train_loss[i+1] += losses[i].item()
            
            optimizer.zero_grad()
            overall_loss.backward()
            optimizer.step()
    
        else:
            model.eval()

            with torch.no_grad():
                 for idx in range(len(valid_batches)):
                    batch = get_current_batch(valid_batches, batch_size, batch_count[1], shuffled_valid)
                    images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, batch_count, 1, transforms, 
                                                            scaled_vad, scaling_type)
                    images, labels = images.to(device), labels.to(device)

                    preds, _, _ = model.forward(images)
                    pred_labels = []
                    
                    for idx, l_type in enumerate(task_type):
                        pred_labels.append(get_labels(labels, idx, l_type).to(device))
                    
                    losses = []
                    for idx, crit in enumerate(criterion):
                        losses.append(crit(preds[idx], pred_labels[idx]))
                    
                    overall_loss = model.compute_loss(losses)
                    
                    epoch_valid_loss[0] += overall_loss.item()
                    for i in range(len(losses)):
                        epoch_valid_loss[i+1] += losses[i].item()
                    
                    inverse_scaler(preds, pred_labels, stats, scaling_type)

                    for idx, metric in enumerate(metrics):
                        if idx in [0, 1]:
                            epoch_metrics[idx] += metric(preds[0], pred_labels[0]).item()
                        elif idx in [2, 3]:
                            epoch_metrics[idx] += metric(preds[1], pred_labels[1]).item()
                           
                    
                    for idx, metric in enumerate(metrics_macro):
                        if idx in [0, 1]:
                            epoch_metrics_macro[idx] += metric(preds[0], pred_labels[0]).item()
                        elif idx in [2, 3]:
                            epoch_metrics_macro[idx] += metric(preds[1], pred_labels[1]).item()
                       
                    for idx, metric in enumerate(metrics_none):
                        if idx == 0:
                          metric_output = metric(preds[0], pred_labels[0])
                          for i, el in enumerate(torch.isnan(metric_output)):
                              if el == 1:
                                  emo_div_amount_none[i] -= 1
                        elif idx == 2:
                          metric_output = metric(preds[1], pred_labels[1])
                          for i, el in enumerate(torch.isnan(metric_output)):
                              if el == 1:
                                  gen_div_amount_none[i] -= 1

                    for idx, metric in enumerate(metrics_none):
                        if idx in [0, 1]:
                            metric_output = metric(preds[0], pred_labels[0])
                            metric_output = torch.nan_to_num(metric_output)
                            epoch_metrics_none[idx] += metric_output
                            
                        elif idx in [2, 3]:
                            metric_output = metric(preds[1], pred_labels[1])
                            metric_output = torch.nan_to_num(metric_output)
                            epoch_metrics_none[idx] += metric_output

                    for idx, metric in enumerate(metrics_num):
                        epoch_metrics_num[idx] += metric(preds[idx+2], pred_labels[idx+2]).item()

                    for idx, conf_matrix in enumerate(confusion_matrix):
                        epoch_confusion[idx] += conf_matrix(preds[idx], pred_labels[idx])

        scheduler.step(epoch_valid_loss[0]/len(valid_batches))
        
        end = time.time()
        epoch_duration.append(end - start)

        training_loss.append([loss/len(train_batches) for loss in epoch_train_loss])
        validation_loss.append([loss/len(valid_batches) for loss in epoch_valid_loss])
        performance_metric.append([metric/len(valid_batches) for metric in epoch_metrics])
        performance_metric_macro.append([metric/len(valid_batches) for metric in epoch_metrics_macro])
        performance_metric_num.append([metric/len(valid_batches) for metric in epoch_metrics_num])
        
        emo_div_amount_none = torch.as_tensor(emo_div_amount_none).to(device)
        gen_div_amount_none = torch.as_tensor(gen_div_amount_none).to(device)
        
        li = []
        for i, metric in enumerate(epoch_metrics_none):
            if i in [0,1]:
                li.append(metric/emo_div_amount_none)
            elif i in [2,3]:
                li.append(metric/gen_div_amount_none)
        epoch_metrics_none = li
        
        batch_count = [0, 0]

        progress_bar.value = e + 1
        if progress_bar.value == epochs[1]:
            progress_bar.description = f'Training Completed'
        else:
            progress_bar.description = f'Epoch - {progress_bar.value} / {epochs[1]}'

        if epoch_valid_loss[0]/len(valid_batches) < min_validation_loss:
            min_validation_loss = epoch_valid_loss[0]/len(valid_batches)
            current_best = {
                'model': model.state_dict(),
                'current epoch': e,
                'plotting data': (training_loss[-1], validation_loss[-1], epoch_duration[-1]),
                'confusion matrix': epoch_confusion,
                'weighted metric': performance_metric[-1], 
                'macro metric': performance_metric_macro[-1], 
                'class metric': epoch_metrics_none,
                'numerical metric': performance_metric_num[-1]
            }
            torch.save(current_best, f'{meta_root}/Best.pt')


    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'current epoch': epochs,
        'next epoch': epochs[1],
        'plotting data': (training_loss, validation_loss, epoch_duration),
        'weighted metric': performance_metric, 
        'macro metric': performance_metric_macro,
        'numerical metric': performance_metric_num
    }
    torch.save(checkpoint, f'{meta_root}/Checkpoint-{checkpoint_num}.pt')


def test_iemocap_network(roots: tuple, model_params: tuple, torch_params: dict, other_params: dict, name: str) -> None:
    '''
    Tests the final network on iemocap

    param: roots = names of root folders
    param: model_params = data related to model
    param: torch_params = data related to pytorch
    param: other_params = data related to other details
    param: name = session number

    return: None
    '''

    # Unpacking data
    root, meta_root = roots
    model, transforms, = model_params
    metrics_num, confusion_matrix, metrics, metrics_macro, metrics_none  = torch_params.values()
    scaling_type, task_type, num_augs = other_params.values()
    
    label_data = load_data(meta_root, 'Label')
    split_data = load_data(meta_root, 'Split')
    class_data = load_data(meta_root, 'Class')
    scaled_vad = load_data(meta_root, 'Scaled VAD')
    stats = load_data(meta_root, 'Statistics')
     
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    for metric in metrics:
        metric.to(device)
    
    for metric in metrics_macro:
        metric.to(device)

    for metric in metrics_none:
        metric.to(device)
    
    for metric in metrics_num:
        metric.to(device)
    
    for matrix in confusion_matrix:
        matrix.to(device)

    split_type = 'test'

    epoch_metrics = [0]*len(metrics)
    epoch_metrics_macro = [0]*len(metrics_macro)
    epoch_metrics_none = [0]*len(metrics_none)
    epoch_metrics_num = [0]*len(metrics_num)
    epoch_confusion = [0]*len(confusion_matrix)

    model.eval() 

    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = 2, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = 'Making predictions...'
    display(progress_bar)

    with torch.no_grad():
        preds, pred_labels = None, None
        for sample_idx, file_name in enumerate(split_data[split_type]):
            batch = get_augmentations([file_name], num_augs)
            images, labels, _ = load_iemocap_batch(root, split_type, label_data, class_data, batch, None, None, transforms, scaled_vad, scaling_type)
            
            images, labels = images.to(device), labels.to(device)

            sample_preds, _, _ = model.forward(images)
            sample_pred_labels = []
          
            for idx, l_type in enumerate(task_type):
                sample_pred_labels.append(get_labels(labels, idx, l_type).to(device))

            sample_preds, sample_pred_labels = ensemble_preds(sample_preds, sample_pred_labels)

            if sample_idx == 0:
              preds = sample_preds
              pred_labels = sample_pred_labels
            
            else:
              for i in range(0, 5):
                preds[i] = torch.cat((preds[i], sample_preds[i]), dim = 0)
                pred_labels[i] = torch.cat((pred_labels[i], sample_pred_labels[i]), dim = 0)

        progress_bar.value += 1
        progress_bar.description = 'Calculating metrics...'

        inverse_scaler(preds, pred_labels, stats, scaling_type)

        for idx, metric in enumerate(metrics):
            if idx in [0, 1]:
                epoch_metrics[idx] += metric(preds[0], pred_labels[0]).item()
            elif idx in [2, 3]:
                epoch_metrics[idx] += metric(preds[1], pred_labels[1]).item()
                    
        for idx, metric in enumerate(metrics_macro):
            if idx in [0, 1]:
                epoch_metrics_macro[idx] += metric(preds[0], pred_labels[0]).item()
            elif idx in [2, 3]:
                epoch_metrics_macro[idx] += metric(preds[1], pred_labels[1]).item()
                       
        for idx, metric in enumerate(metrics_none):
            if idx in [0, 1]:
                epoch_metrics_none[idx] += metric(preds[0], pred_labels[0])    
            elif idx in [2, 3]:
                epoch_metrics_none[idx] += metric(preds[1], pred_labels[1])

        for idx, metric in enumerate(metrics_num):
            epoch_metrics_num[idx] += metric(preds[idx+2], pred_labels[idx+2]).item()

        for idx, conf_matrix in enumerate(confusion_matrix):
            epoch_confusion[idx] += conf_matrix(preds[idx], pred_labels[idx])

        progress_bar.value += 1
        progress_bar.description = 'Testing Completed'

    test_metrics = {
            'confusion matrix': epoch_confusion,
            'weighted metric': epoch_metrics, 
            'macro metric': epoch_metrics_macro, 
            'class metric': epoch_metrics_none,
            'numerical metric': epoch_metrics_num
        }
    torch.save(test_metrics, f'{meta_root}/Test-{name}.pt')


def get_single_iemocap(roots: tuple, model, split_type: str, plot_params: tuple) -> tuple:
    '''
    Obtains the prediction of a single iemocap sample

    param: roots = root folder names
    param: model = trained model
    param: split_type = type of split
    param: plot_params = additional parameters 

    return: predictions of sample
    '''
  
    root, meta_root = roots
    task_type, aug_type, transforms, scaling_type = plot_params

    split_data = load_data(meta_root, 'Split')
    label_data = load_data(meta_root, 'Label')
    class_data = load_data(meta_root, 'Class')
    scaled_vad = load_data(meta_root, 'Scaled VAD')
    stats = load_data(meta_root, 'Statistics')

    file_name = np.random.choice(split_data[split_type])
    batch = get_augmentations([file_name], num_augs = [aug_type])
    images, labels, _ = load_iemocap_batch(root, 'test', label_data, class_data, batch, None, None, transforms, scaled_vad, scaling_type)
    preds, _, _ = model.forward(images)
    
    pred_labels = []
    for idx, l_type in enumerate(task_type):
        pred_labels.append(get_labels(labels, idx, l_type))
    
    inverse_scaler(preds, pred_labels, stats, scaling_type)

    return (preds, pred_labels, file_name, images)


def get_single_ravdess(roots: tuple, model, split_type: str, plot_params: tuple) -> tuple:
    '''
    Obtains the prediction of a single ravdess sample

    param: roots = root folder names
    param: model = trained model
    param: split_type = type of split
    param: plot_params = additional parameters 

    return: predictions of sample
    '''

    root, rav_meta_root, iem_meta_root = roots
    task_type, aug_type, transforms = plot_params

    split_data = load_data(rav_meta_root, 'Split')
    label_data = load_data(rav_meta_root, 'Label')
    class_data = load_data(rav_meta_root, 'Class')

    stats = load_data(iem_meta_root, 'Statistics')

    file_name = np.random.choice(split_data[split_type])
    batch = get_augmentations([file_name], num_augs = [aug_type])
    images, labels, _ = load_ravdess_batch(root, 'test', label_data, class_data, batch, transforms)
    preds, _, _ = model.forward(images)
    
    pred_labels = []
    for idx, l_type in enumerate(task_type):
        pred_labels.append(get_labels(labels, idx, l_type))
    

    inverse_scaler(preds, None, stats, stats['scaling type'])
    map_intensity(preds)
    remove_predictions(preds)

    return (preds, pred_labels, file_name, images)


def get_iemocap_classwise(roots: tuple, model, plot_params: tuple, num_samples: int) -> tuple:
    '''
    Obtains the predictions and organizes them by class (for emotion and gender)  

    param: roots = root folder names
    param: model = trained model
    param: plot_params = additional parameters 
    param: num_samples = number of predictions

    return: classwise predictions
    '''
  
    root, meta_root = roots
    task_type, aug_type, transforms, scaling_type = plot_params

    split_data = load_data(meta_root, 'Split')
    class_data = load_data(meta_root, 'Class')
    label_data = load_data(meta_root, 'Label')
    scaled_vad = load_data(meta_root, 'Scaled VAD')

    emotion, gender = {}, {}
    for key in list(class_data.values())[:-2]:
        emotion[key] = []
    for key in list(class_data.values())[-2:]:
        gender[key] = []

    indices = torch.randperm(len(split_data['test']))
    indices = indices[:num_samples]

    for idx in indices:
        file_name = split_data['test'][idx]
        batch = get_augmentations([file_name], num_augs = [aug_type])
        images, labels, _ = load_iemocap_batch(root, 'test', label_data, class_data, batch, None, None, transforms, scaled_vad, scaling_type)
        preds, _, _ = model.forward(images)

        pred_labels = []
        for idx, l_type in enumerate(task_type):
            pred_labels.append(get_labels(labels, idx, l_type))

        for i in range(0, 2):
            preds[i] = torch.exp(preds[i])
    
        emotion[int(pred_labels[0][0])].append(preds[0][0][int(pred_labels[0][0])].item())
        gender[int(pred_labels[1][0])].append(preds[1][0][int(pred_labels[1][0])].item())
    
    return (emotion, gender)


def get_ravdess_classwise(roots: tuple, model, plot_params: tuple, num_samples: int) -> tuple:
    '''
    Obtains the predictions and organizes them by class (for emotion and gender)  

    param: roots = root folder names
    param: model = trained model
    param: plot_params = additional parameters 
    param: num_samples = number of predictions

    return: classwise predictions
    '''

    root, meta_root, _ = roots
    task_type, aug_type, transforms = plot_params

    split_data = load_data(meta_root, 'Split')
    class_data = load_data(meta_root, 'Class')
    label_data = load_data(meta_root, 'Label')

    emotion, gender = {}, {}
    for key in list(class_data.values())[:-4]:
        emotion[key] = []
    for key in list(class_data.values())[-4:-2]:
        gender[key] = []

    indices = torch.randperm(len(split_data['test']))
    indices = indices[:num_samples]

    for idx in indices:
        file_name = split_data['test'][idx]
        batch = get_augmentations([file_name], num_augs = [aug_type])
        images, labels, _ = load_ravdess_batch(root, 'test', label_data, class_data, batch, transforms)
        preds, _, _ = model.forward(images)

        pred_labels = []
        for idx, l_type in enumerate(task_type):
            pred_labels.append(get_labels(labels, idx, l_type))

        for i in range(0, 2):
            preds[i] = torch.exp(preds[i])
    
        emotion[int(pred_labels[0][0])].append(preds[0][0][int(pred_labels[0][0])].item())
        gender[int(pred_labels[1][0])].append(preds[1][0][int(pred_labels[1][0])].item())
    
    return (emotion, gender)
  

def map_intensity(preds: list) -> None:
    '''
    Maps the activation prediction to an intensity label (threshold is set to 3.5)

    param: preds = predictions

    return: None
    '''
    
    intensity_softmax = None
    
    for value_idx in range(preds[3].shape[0]):
        value = preds[3][value_idx][0]
        
        softmax_val = [[0.0, 0.0]]
        idx = 0 if value.item() < 3.5 else 1
        softmax_val[0][idx] = 1.0

        softmax_val = torch.as_tensor(softmax_val)
        softmax_val = torch.log(softmax_val)

        if value_idx == 0:
            intensity_softmax = softmax_val
        else:
            intensity_softmax = torch.cat((intensity_softmax, softmax_val), dim = 0)

    activation = None
    for i in range(3):
        if i == 1:
            activation = preds.pop(2)
        else:
            preds.pop(2)

    preds.append(intensity_softmax)
    preds.append(activation)


def remove_predictions(preds: list) -> None:
    '''
    Removes the predictions of certain emotion classes

    param: preds = predictions

    return: None
    '''

    to_remove = [1, 2]
    new_preds = []

    for idx, prediction in enumerate(preds[0]):
        for i in to_remove:
            prediction = torch.cat([prediction[0:i], prediction[i+1:]])
        
        prediction = torch.as_tensor([prediction.tolist()])
     
        if idx == 0:
            new_preds = prediction
        else:
            new_preds = torch.cat((new_preds, prediction), dim = 0)

    preds[0] = new_preds


def test_ravdess_network(roots: tuple, model_params: tuple, torch_params: dict, other_params: dict, name: str) -> None:
    '''
    Tests the final network on ravdess

    param: roots = names of root folders
    param: model_params = data related to model
    param: torch_params = data related to pytorch
    param: other_params = data related to other details
    param: name = session number

    return: None
    '''

    # Unpacking data
    root, rav_meta_root, iem_meta_root = roots
    model, transforms = model_params
    confusion_matrix, metrics, metrics_macro, metrics_none  = torch_params.values()
    scaling_type, task_type, num_augs = other_params.values()
    
    label_data = load_data(rav_meta_root, 'Label')
    split_data = load_data(rav_meta_root, 'Split')
    class_data = load_data(rav_meta_root, 'Class')
    
    stats = load_data(iem_meta_root, 'Statistics')
     
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    for metric in metrics:
        metric.to(device)
    
    for metric in metrics_macro:
        metric.to(device)

    for metric in metrics_none:
        metric.to(device)
    
    for matrix in confusion_matrix:
        matrix.to(device)

    split_type = 'test'

    epoch_metrics = [0]*len(metrics)
    epoch_metrics_macro = [0]*len(metrics_macro)
    epoch_metrics_none = [0]*len(metrics_none)
    epoch_confusion = [0]*len(confusion_matrix)

    model.eval() 

    layout = Layout(width = '500px')
    progress_bar = IntProgress(min = 0, max = 2, style = {'description_width': 'initial'}, layout = layout)
    progress_bar.description = 'Making predictions...'
    display(progress_bar)

    with torch.no_grad():
        preds, pred_labels = None, None
        for sample_idx, file_name in enumerate(split_data[split_type]):
            batch = get_augmentations([file_name], num_augs)
            images, labels, _ = load_ravdess_batch(root, split_type, label_data, class_data, batch, transforms)
            
            images, labels = images.to(device), labels.to(device)

            sample_preds, _, _ = model.forward(images)
            sample_pred_labels = []
          
            for idx, l_type in enumerate(task_type):
                sample_pred_labels.append(get_labels(labels, idx, l_type).to(device))
            


            sample_preds, sample_pred_labels = ensemble_preds(sample_preds, sample_pred_labels, name = 'ravdess')

            if sample_idx == 0:
              preds = sample_preds
              pred_labels = sample_pred_labels
            
            else:
              for i in range(0, 5):
                preds[i] = torch.cat((preds[i], sample_preds[i]), dim = 0)
              
              for i in range(0, 3):
                pred_labels[i] = torch.cat((pred_labels[i], sample_pred_labels[i]), dim = 0)

        progress_bar.value += 1
        progress_bar.description = 'Calculating metrics...'

        inverse_scaler(preds, None, stats, scaling_type)
        map_intensity(preds)
        remove_predictions(preds)

        for idx, metric in enumerate(metrics):
            if idx in [0, 1]:
                epoch_metrics[idx] += metric(preds[0], pred_labels[0]).item()
            elif idx in [2, 3]:
                epoch_metrics[idx] += metric(preds[1], pred_labels[1]).item()
            elif idx in [4, 5]:
                epoch_metrics[idx] += metric(preds[2], pred_labels[2]).item()
                    
        for idx, metric in enumerate(metrics_macro):
            if idx in [0, 1]:
                epoch_metrics_macro[idx] += metric(preds[0], pred_labels[0]).item()
            elif idx in [2, 3]:
                epoch_metrics_macro[idx] += metric(preds[1], pred_labels[1]).item()
            elif idx in [4, 5]:
                epoch_metrics_macro[idx] += metric(preds[2], pred_labels[2]).item()
                       
        for idx, metric in enumerate(metrics_none):
            if idx in [0, 1]:
                epoch_metrics_none[idx] += metric(preds[0], pred_labels[0])
            elif idx in [2, 3]:
                epoch_metrics_none[idx] += metric(preds[1], pred_labels[1])
            elif idx in [4, 5]:
                epoch_metrics_none[idx] += metric(preds[2], pred_labels[2])

        for idx, conf_matrix in enumerate(confusion_matrix):
            epoch_confusion[idx] += conf_matrix(preds[idx], pred_labels[idx])

        progress_bar.value += 1
        progress_bar.description = 'Testing Completed'

    test_metrics = {
            'confusion matrix': epoch_confusion,
            'weighted metric': epoch_metrics, 
            'macro metric': epoch_metrics_macro, 
            'class metric': epoch_metrics_none
        }
    torch.save(test_metrics, f'{rav_meta_root}/Test-{name}.pt')

