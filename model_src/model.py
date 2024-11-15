from torch import nn

class BoxingMatchPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BoxingMatchPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

