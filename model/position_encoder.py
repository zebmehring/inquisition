import torch


class PositionEncoder(torch.nn.Module):
    def __init__(self, num_positions):
        super(PositionEncoder, self).__init__()
        positions = torch.arange(num_positions)
        frequencies = torch.pow(10000, 2 * ???)
        self.position_encodings = torch.zeros(num_positions)
        self.position_encodings[0::2] = torch.sin(positions / frequencies)
        self.position_encodings[1::2] = torch.cos(positions / frequencies)

    def forward(self, x):
        return self.position_encodings + x
