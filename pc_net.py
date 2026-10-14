import math
import torch

class PredictiveCodingNetwork:
    def __init__(self, layer_ns=[784, 128, 64, 10], lr=0.001, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.layer_ns = layer_ns
        self.n_layers = len(layer_ns) - 1

        self.weights = []
        self.biases = []
        for l in range(self.n_layers):
            din = layer_ns[l]
            dout = layer_ns[l + 1]
            W = torch.randn(din, dout, device=self.device) * math.sqrt(2.0 / din)
            b = torch.zeros(dout, device=self.device) 
            self.weights.append(W)
            self.biases.append(b)

        self.params = self.weights + self.biases
        self.optimizer = torch.optim.Adam(self.params, lr=lr)

    def to(self, device):
        self.device = torch.device(device)
        for l in range(self.n_layers):
            self.weights[l] = self.weights[l].to(self.device)
            self.biases[l] = self.biases[l].to(self.device)
        self.params = self.weights + self.biases
        return self

    def predict_layer(self, z_in, l):
        pre = z_in @ self.weights[l] + self.biases[l]
        if l < self.n_layers - 1:
            return torch.relu(pre)
        return pre

    def forward(self, x, return_states=False):
        curr = x.view(x.size(0), -1)
        states = [curr]
        for l in range(self.n_layers):
            curr = self.predict_layer(curr, l)
            states.append(curr) 
        if return_states:
            return states
        return curr

    def get_errors(self, z):
        mu = []
        eps = []
        delta = []
        for l in range(self.n_layers):
            pred = self.predict_layer(z[l], l) 
            err = z[l + 1] - pred
            if l < self.n_layers - 1:
                d = err * (pred > 0).float()
            else:
                d = err
            mu.append(pred)
            eps.append(err)
            delta.append(d)
        return mu, eps, delta

    def infer_and_learn(self, x, y, n_inference_steps=20, lr_z=0.01):
        B = x.size(0)
        z = self.forward(x, return_states=True)
        z[-1] = y.clone()
        
        energy_history = []
        for t in range(n_inference_steps):
            mu, eps, delta = self.get_errors(z)

            layer_energies = []
            for e in eps:
                layer_energies.append(0.5 * torch.sum(e ** 2).item() / B)
            total_energy = sum(layer_energies)

            energy_history.append({
                "step": t,
                "layer_energies": layer_energies,
                "total_energy": total_energy
            })

            new_z = [z[0]]
            for k in range(1, self.n_layers):
                self_error = eps[k - 1] 
                feedback = delta[k] @ self.weights[k].T
                updated_zk = z[k] - lr_z * (self_error - feedback)
                new_z.append(updated_zk)
            new_z.append(z[-1])
            z = new_z

        _, _, delta = self.get_errors(z)
        for l in range(self.n_layers):
            self.weights[l].grad = - (z[l].T @ delta[l]) / B
            self.biases[l].grad = - delta[l].sum(dim=0) / B

        torch.nn.utils.clip_grad_norm_(self.params, max_norm=1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

        return energy_history

    def evaluate(self, test_loader):
        total_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                preds = self.forward(images)
                targets = torch.eye(self.layer_ns[-1], device=self.device)[labels]
                total_loss += torch.mean((targets - preds) ** 2).item()
                correct += (preds.argmax(dim=1) == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(test_loader), 100.0 * correct / total
