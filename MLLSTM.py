import torch
import torch.nn as nn


class MLLSTM(nn.Module):
    def __init__(self, input_size):
        super(MLLSTM, self).__init__()

        # Define LSTM layers
        self.lstm_layer1 = nn.LSTM(input_size=input_size, hidden_size=64, batch_first=True)
        self.lstm_layer2 = nn.LSTM(input_size=65, hidden_size=64, batch_first=True)  # Concatenate 64 + 1
        self.lstm_layer3 = nn.LSTM(input_size=65, hidden_size=64, batch_first=True)  # Concatenate 64 + 1
        self.lstm_layer4 = nn.LSTM(input_size=89, hidden_size=64, batch_first=True)  # Concatenate 64 + 25

        # Define fully connected layer
        self.fc_layer = nn.Linear(27, 64)

        # Define output layer
        self.output_layer = nn.Linear(64, 2)

    def forward(self, input_data):
        # Reshape input_data to fit LSTM
        batch_size, num_channels, seq_len, num_feats = input_data.size()
        input_data_reshaped = input_data.view(batch_size, seq_len, num_feats * num_channels)

        # Split the input into different parts
        input_time = input_data_reshaped[:, :, 0:1]
        input_freq = input_data_reshaped[:, :, 1:2]
        input_feat = input_data_reshaped[:, :, 2:3]
        input_timefreq = input_data_reshaped[:, :, 3:]  # This assumes input_timefreq has 24 features

        # LSTM layer 1
        lstm_out1, _ = self.lstm_layer1(input_time)
        
        # Concatenate frequency input with LSTM layer 1 output
        lstm_out1_cat = torch.cat((lstm_out1, input_freq), dim=2)
        
        # LSTM layer 2
        lstm_out2, _ = self.lstm_layer2(lstm_out1_cat)

        # Concatenate feature input with LSTM layer 2 output
        lstm_out2_cat = torch.cat((lstm_out2, input_feat), dim=2)
        
        # LSTM layer 3
        lstm_out3, _ = self.lstm_layer3(lstm_out2_cat)
        
        # LSTM layer 4
        lstm_out4_cat = torch.cat((lstm_out3, input_timefreq), dim=2)
        lstm_out4, _ = self.lstm_layer4(lstm_out4_cat)
        
        # Apply fully connected layer
        fc_out = self.fc_layer(lstm_out4[:, :, -1])  # Taking the last timestep output
        
        # Output layer
        output = torch.sigmoid(self.output_layer(fc_out))

        return output

