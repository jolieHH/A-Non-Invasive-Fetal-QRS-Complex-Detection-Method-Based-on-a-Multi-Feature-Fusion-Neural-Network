import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class DeepLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x1 = x[:,:,0:1,:]  # The first 2 rows in the dataset are time-domain pulse
        x1 = x1.view(x1.shape[0], 1, -1)
        lstm_out, _ = self.lstm(x1)
        lstm_out = lstm_out[:, -1, :]
        output = self.fc(lstm_out)
        return output

class DeepFC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DeepFC, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.fc5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        x = x[:,:,0:1,:].view(x.size(0), -1)  # Reshape to (batch_size, input_size)
        x = F.relu(self.fc1(x))  # Fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc2(x))  # Fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc3(x))  # Fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc4(x))  # Fully connected layer with ReLU activation
        x = self.dropout(x)  # Apply dropout
        x = self.fc5(x)  # Fully connected layer
        return x


class SimpleCNN(nn.Module):
    def __init__(self,output_size):
        super(SimpleCNN, self).__init__()
        self.flatten = nn.Flatten()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(416, 64)#256
        self.fc2 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x2 = x[:,:,1:2,:]  # The third row in the dataset is frequency-domain pulse
        x2 = x2.view(x2.shape[0], 1, -1)
        x2 = self.conv1(x2)
        x2 = self.relu(x2)
        x2 = self.pool(x2)
        x2 = self.conv2(x2)
        x2 = self.relu(x2)
        x2 = self.pool(x2)
        x2 = x2.view(x2.shape[0],-1)
        x2 = self.fc1(x2)
        x2 = self.relu(x2)
        x2 = self.fc2(x2)
        return x2

class DNN4(nn.Module):
    def __init__(self,output_size):
        super(DNN4, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(60, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 64)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(64, output_size)
        
    def forward(self, x):
        x3 = x[:,:,2:3,:]  # The fourth row in the dataset is features
        x3 = self.flatten(x3)
        x3 = self.relu1(self.fc1(x3))
        x3 = self.relu2(self.fc2(x3))
        x3 = self.relu3(self.fc3(x3))
        x3 = self.fc4(x3)
        return x3

class ShuffleNetBinary(nn.Module):
    def __init__(self, output_size):
        super(ShuffleNetBinary, self).__init__()

        # Load the ShuffleNet model
        self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

        # Modify the classifier to match the output_size
        in_features = self.shufflenet.fc.in_features
        self.shufflenet.fc = nn.Linear(in_features, output_size)

    def forward(self, x):
        x4 = x[:, :, 0:27, :]  # The fifth to 38th rows are time-frequency data
        x4 = torch.cat([x4] * 3, dim=1)
        return self.shufflenet(x4)

# class ShuffleNetBinary(nn.Module):
#     def __init__(self, output_size):
#         super(ShuffleNetBinary, self).__init__()

#         # Load the ShuffleNet model
#         self.shufflenet = models.shufflenet_v2_x1_0(pretrained=True)

#         # Modify the classifier to match the output_size
#         in_features = self.shufflenet.fc.in_features
#         self.shufflenet.fc = nn.Linear(in_features, output_size)

#     def forward(self, x, ablation_type=None):
#         if ablation_type == 'remove_first':
#             x4 = x[:, :, 1:27, :]  # Remove the first row
#         elif ablation_type == 'remove_second':
#             x4 = torch.cat([x[:, :, 0:1, :], x[:, :, 2:27, :]], dim=2)  # Remove the second row
#         elif ablation_type == 'remove_third':
#             x4 = torch.cat([x[:, :, 0:2, :], x[:, :, 3:27, :]], dim=2)  # Remove the third row
#         elif ablation_type == 'remove_fourth_to_27th':
#             x4 = x[:, :, 0:3, :]  # Remove rows 4 to 27
#         else:
#             x4 = x[:, :, 0:27, :]  # Use all rows (default case)
        
#         x4 = torch.cat([x4] * 3, dim=1)
#         return self.shufflenet(x4)

# Example usage:
# model = ShuffleNetBinary(output_size=2)
# output = model(input_data, ablation_type='remove_first')


class ResNetBinary(nn.Module):
    def __init__(self, output_size):
        super(ResNetBinary, self).__init__()

        # Load the ResNet model
        self.resnet = models.resnet34()

        # Modify the classifier to match the output_size
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, output_size)
 
        # Adjust the first convolution layer to accept 1 input channel
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=3, bias=False)

    def forward(self, x, ablation_type='none'):#none
        if ablation_type == 'remove_first':
            x4 = x[:, :, 1:27, :]  # Remove the first row
        elif ablation_type == 'remove_second':
            x4 = torch.cat([x[:, :, 0:1, :], x[:, :, 2:27, :]], dim=2)  # Remove the second row
        elif ablation_type == 'remove_third':
            x4 = torch.cat([x[:, :, 0:2, :], x[:, :, 3:27, :]], dim=2)  # Remove the third row
        elif ablation_type == 'remove_fourth_to_27th':
            x4 = x[:, :, 0:3, :]  # Remove rows 4 to 27
        else:
            x4 = x[:, :, 0:27, :]  # Use all rows (default case)
          
        return self.resnet(x4)

class MultiModalAttentionModel(nn.Module):
    def __init__(self, lstm_model, cnn_model, dnn_model, ShuffleNet_model, attention_heads, output_size):
        super(MultiModalAttentionModel, self).__init__()

        # 1. Define the individual models
        self.lstm_model = lstm_model
        self.cnn_model = cnn_model
        self.dnn_model = dnn_model
        self.ShuffleNet_model = ShuffleNet_model

        # 2. Define the attention layer
        self.attention = nn.MultiheadAttention(embed_dim=output_size, num_heads=attention_heads)

        # 3. Define the final classification layer
        self.fc_final = nn.Linear(output_size, 2)

    def forward(self, x):
        # Forward pass through each individual model
        lstm_output = self.lstm_model(x)
        cnn_output = self.cnn_model(x)
        dnn_output = self.dnn_model(x)
        ShuffleNet_output = self.ShuffleNet_model(x)

        # Concatenate the outputs along the feature dimension
        concatenated_output = torch.cat((lstm_output, cnn_output, dnn_output, ShuffleNet_output), dim=1)

        # Apply attention to the concatenated output
        attention_output, _ = self.attention(concatenated_output, concatenated_output, concatenated_output)

        # Final classification layer
        final_output = self.fc_final(attention_output)

        return final_output
