import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, channel_size, keypoint_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=channel_size * keypoint_size,
                            hidden_size=channel_size * keypoint_size,
                            num_layers=num_layers,
                            batch_first=True, bidirectional=False)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        B, C, T, V = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(B, T, C * V)

        x, hidden_state = self.lstm(x)

        x = x.view(B, T, C, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        return x


if __name__ == '__main__':
    x = torch.randn(64, 96, 300, 25)
    lstm = LSTM(channel_size=96, keypoint_size=25, num_layers=4)
    output = lstm(x)
    print(output.size())
