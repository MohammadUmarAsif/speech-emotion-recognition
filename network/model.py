import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleNetwork(nn.Module):
    def __init__(self, model_params: dict) -> None:
        '''
        Constructor for the class

        param: model_params = parameters for the model

        return: None
        '''
        
        super(SimpleNetwork, self).__init__()
        
        self.conv_list = nn.ModuleList([])
        self.pool_list = nn.ModuleList([])
        self.hidden_list = nn.ModuleList([])
        self.pool_position = model_params['pool position']

        for param in model_params['convolutional']:
            self.conv_list.extend([nn.Conv2d(param['input'], param['output'], kernel_size = param['kernel'],
                                    stride = param['stride'], padding = param['padding'])])
            
        for param in model_params['pooling']:
            self.pool_list.extend([nn.MaxPool2d(param['kernel'], stride = param['stride'])])
        
        for param in model_params['hidden']:
            self.hidden_list.extend([nn.Linear(param['input'], param['output'])])
 
        self.output_layer = nn.Linear(model_params['hidden'][-1]['output'], model_params['output'])

    def forward(self, x: torch.Tensor) -> list:
        '''
        Forward pass through the network

        param: x = batch of images

        return: predictions
        '''
        
        pool_idx = 0
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            x = F.relu(x)

            if (pool_idx < len(self.pool_position)) and (self.pool_position[pool_idx] - 1 == i):
                x = self.pool_list[pool_idx](x)
                pool_idx += 1

        x = x.view(x.shape[0], -1)
        
        for i in range(len(self.hidden_list)):
            x = self.hidden_list[i](x)
            x = F.relu(x)
    
        preds = F.log_softmax(self.output_layer(x), dim = 1)

        return preds


class ComplexNetwork(nn.Module):
    def __init__(self, model_params: dict, layer_params: dict, multitask_params: dict, task_type: tuple) -> None:
        '''
        Constructor for the class

        param: model_params = parameters for the model
        param: layer_params = parameters for model layers
        param: multitask_params = parameters for multitask learning
        param: weight_init = if weights should be initialized

        return: None
        '''

        super(ComplexNetwork, self).__init__()
        
        self.conv_list = nn.ModuleList([])
        self.batch_list = None
        self.pool_list = nn.ModuleList([])
        self.hidden_list = nn.ModuleList([])
        self.task_list = None
        self.output_list = nn.ModuleList([])
        self.pool_position = model_params['pool position']

        activation, pool_type, dropout, batch_norm, attention, weight_init = layer_params.values()
        loss_type, loss_weights, task_params = multitask_params.values()

        self.label_type = task_type
        self.loss_type = loss_type
        self.task_params = task_params

        self.dropout = nn.Dropout(p = dropout)
        self.lambda_param = attention

        self.batch_norm = batch_norm
        if batch_norm:
            self.batch_list = nn.ModuleList([])

        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'leaky_relu':
            self.activation = F.leaky_relu

        self.pool_type = None
        if pool_type == 'max':
            self.pool_type = nn.MaxPool2d
        elif pool_type == 'average':
            self.pool_type = nn.AvgPool2d

        if loss_type == 'weighted':
            self.loss_weights = loss_weights
        elif loss_type == 'param':
            self.log_vars = nn.Parameter(torch.zeros((len(model_params['output']))))
        elif loss_type == 'weighted param':
            self.log_vars = nn.Parameter(torch.zeros((len(model_params['output']) - 1)))

        for param in model_params['convolutional']:
            self.conv_list.extend([nn.Conv2d(param['input'], param['output'], kernel_size = param['kernel'],
                                    stride = param['stride'], padding = param['padding'])])

            if batch_norm:
                self.batch_list.extend([nn.BatchNorm2d(param['output'])])
            
        for param in model_params['pooling']:
            self.pool_list.extend([self.pool_type(param['kernel'], stride = param['stride'])])
        
        for param in model_params['hidden']:
            self.hidden_list.extend([nn.Linear(param['input'], param['output'])])


        if task_params:
            self.task_list = nn.ModuleList([])
            for param in task_params:
                input_param = model_params['hidden'][-1]['output']
                self.task_list.extend([nn.Linear(input_param, param)])
            
            for idx, param in enumerate(model_params['output']):
                input_param = task_params[idx]
                self.output_list.extend([nn.Linear(input_param, param)])
            
        else:
            for param in model_params['output']:
                input_param = model_params['hidden'][-1]['output']
                self.output_list.extend([nn.Linear(input_param, param)])

        if weight_init:
            self.initialize_layers(task_params, batch_norm, activation)
            
    def forward(self, x: torch.Tensor) -> list:
        '''
        Forward pass through the network

        param: x = batch of images

        return: predictions
        '''
        feature_maps = []
        att_wt = None
        att_out = None
        
        pool_idx = 0
        for i in range(len(self.conv_list)):
            x = self.conv_list[i](x)
            feature_maps.append(x.detach().clone())
            
            if self.batch_norm:
                x = self.batch_list[i](x)

            x = self.activation(x)

            if (pool_idx <= len(self.pool_position) - 1) and (self.pool_position[pool_idx] - 1 == i):
                x = self.pool_list[pool_idx](x)
                pool_idx += 1

        if self.lambda_param is not None:
            x, att_wt = self.attention(x)
            att_out = x.detach().clone()

        x = x.view(x.shape[0], -1)

        x = self.dropout(x)

        for i in range(len(self.hidden_list)):
            x = self.hidden_list[i](x)
            x = self.activation(x)
            x = self.dropout(x)

        preds = []

        if self.task_params:
            for i in range(len(self.task_list)):
                temp_x = self.task_list[i](x)
                temp_x = self.activation(temp_x)
                temp_x = self.dropout(temp_x)
                temp_x = self.output_list[i](temp_x)

                if self.label_type[i] == 'classification':
                    output = F.log_softmax(temp_x, dim = 1)
                    preds.append(output)
               
                elif self.label_type[i] == 'regression':
                    output = self.activation(temp_x)
                    preds.append(output)
                    
        else:
            for i in range(len(self.output_list)):
                temp_x = self.output_list[i](x)

                if self.label_type[i] == 'classification':
                    output = F.log_softmax(temp_x, dim = 1)
                    preds.append(output)
               
                elif self.label_type[i] == 'regression':
                    output = self.activation(temp_x)
                    preds.append(output)

        return (preds, (att_out, att_wt), feature_maps)

    def compute_loss(self, losses: list) -> torch.tensor:
        '''
        Computes the overall loss of network

        param: losses = list of losses

        return: overall loss
        '''
        
        overall_loss = 0

        if not self.loss_type:
            overall_loss += losses[0]
        
        elif self.loss_type == 'sum':
            for loss in losses:
                overall_loss += loss
        
        elif self.loss_type == 'weighted':
            overall_loss += losses[0]
            for idx, weight in enumerate(self.loss_weights):
                overall_loss += weight*losses[idx+1]
            
        elif self.loss_type == 'param':
            for idx, var in enumerate(self.log_vars):
                precision = torch.exp(-var)
                loss = precision*losses[idx] + var
                overall_loss += loss

        elif self.loss_type == 'weighted param':
            weighted_vars = F.softmax(self.log_vars, dim = 0)
            overall_loss += losses[0]
            for idx, var in enumerate(weighted_vars):
                precision = torch.exp(-var)
                loss = precision*losses[idx+1] + var
                overall_loss += loss
                
        return overall_loss

    def initialize_layers(self, task_params: list, batch_norm: bool, activation: str) -> None:
        '''
        Performs weight initialization for all layers of network

        param: task_params = neurons of task-specific layers
        param: batch_norm = if batch normalization layers are required
        param: activation = activation function

        return: None
        '''
        
        for layer in self.conv_list:
            self.initialize_weights(layer, activation)
            
        for layer in self.hidden_list:
            self.initialize_weights(layer, activation)
            
        for layer in self.output_list:
            self.initialize_weights(layer, activation)

        if task_params:
            for layer in self.task_list:
                self.initialize_weights(layer, activation)
                
        if batch_norm:
            for layer in self.batch_list:
                self.initialize_weights(layer, activation)

    def initialize_weights(self, layer, activation: str) -> None:
        '''
        Initializes weights of a layer    

        param: layer = layer of the network
        param: activation = activation function

        return: None
        '''
        
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight.data, nonlinearity = activation)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)

        elif isinstance(layer, nn.BatchNorm2d):
            nn.init.constant_(layer.weight.data, 1)
            if layer.bias is not None:
                nn.init.constant_(layer.bias.data, 0)

    def attention(self, X: torch.tensor) -> tuple:
        '''
        Implements attention for all feature maps

        param: X = feature maps

        return: transformed feature maps and weights
        '''

        spatial_size = X.shape[2] * X.shape[3] - 1
        mu = X.mean(dim=[2,3]).view(X.shape[0], X.shape[1], -1)
        t = X.view(X.shape[0], X.shape[1], -1)
        
        d = (t - mu).pow(2)
        d = d.view(d.shape[0], d.shape[1], X.shape[2], X.shape[3])

        variance = d.sum(dim=[2,3]) / spatial_size
        
        variance = variance.view(X.shape[0], X.shape[1], -1)
        d = d.view(d.shape[0], d.shape[1], -1)

        E_inv = d / (4 * (variance + self.lambda_param)) + 0.5
        E_inv = E_inv.view(E_inv.shape[0], E_inv.shape[1], X.shape[2], X.shape[3])

        sigmoid = torch.sigmoid(E_inv)

        return (X * sigmoid, sigmoid)

    