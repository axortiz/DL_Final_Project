from torch import nn

class BoxingMatchPredictor(nn.Module):
    def __init__(self, input_size, num_classes):
        super(BoxingMatchPredictor, self).__init__()
        
        # Wider layers with more gradual size reduction
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced dropout
            
            nn.Linear(64, num_classes)
        )
        
        # Modified weight initialization
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Using Xavier/Glorot initialization instead of Kaiming
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.network(x)

