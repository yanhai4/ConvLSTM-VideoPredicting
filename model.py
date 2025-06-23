import torch
import torch.nn as nn

from torch.autograd import Variable

from layer.convlstm import ConvLSTM
from layer.cbam import CBAM


class MyModel(nn.Module):
    def __init__(self, hidden_dim, length):
        super(MyModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = len(self.hidden_dim)
        self.length = length
        self.encoder = ConvLSTM(input_dim=1,
                                hidden_dim=self.hidden_dim,
                                kernel_size=(3, 3),
                                num_layers=self.num_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=True)
        self.out_channel = sum(self.hidden_dim)
        self.decoder = nn.Sequential(
            CBAM(gate_channels=self.out_channel),
            nn.ConvTranspose2d(self.out_channel, 16, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1, bias=False)
        )


    def forward(self, input):
        y, _ = self.encoder(input)
        y = torch.cat(y, dim=2)
        output = []
        for i in range(self.length):
            output.append(self.decoder(y[:, i, ...]))
        output = torch.stack(output, dim=1)
        return output


def test():
    model = MyModel(hidden_dim=[8, 16, 64], length=3)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    x = Variable(torch.FloatTensor(torch.rand(4, 3, 1, 5, 5)), requires_grad=True).to(device)
    y = model(x)
    print(y.shape)




