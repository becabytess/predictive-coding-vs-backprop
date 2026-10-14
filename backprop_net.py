import torch
import torch.nn as nn

class BackpropMLP(nn.Module):
    def __init__(self, layer_ns=[784, 128, 64, 10]): 
        super().__init__()
        self.layer_ns = layer_ns
        layers = [] 

        n_layers = len(layer_ns)
        for i in range(n_layers - 1):
            layers.append(nn.Linear(layer_ns[i], layer_ns[i + 1]))
            if i < n_layers - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.net(x.view(x.size(0), -1))

    def evaluate(self, test_loader):
        device = next(self.parameters()).device
        self.eval() 
        criterion = nn.MSELoss()
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.forward(images)
                targets = torch.eye(self.layer_ns[-1], device=device)[labels]
                total_loss += criterion(outputs, targets).item()
                correct += (outputs.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(test_loader), 100.0 * correct / total
