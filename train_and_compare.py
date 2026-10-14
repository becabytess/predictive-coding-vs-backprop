import time
import tracemalloc
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from data import get_mnist_loaders, to_one_hot
from backprop_net import BackpropMLP
from pc_net import PredictiveCodingNetwork

def run_experiment(epochs=5, batch_size=64):
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_mnist_loaders(batch_size)

    bp_model = BackpropMLP(layer_ns=[784, 128, 64, 10]).to(device)
    params = sum(p.numel() for p in bp_model.parameters())
    print(f"Model Parameters: {params:,} | Device: {device.type.upper()}\n")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    else:
        tracemalloc.start()

    bp_start = time.time()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(bp_model.parameters(), lr=0.002)

    bp_train_losses = []
    bp_val_losses = []
    bp_val_accs = []

    for epoch in range(1, epochs + 1):
        bp_model.train()
        total_loss = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            targets = to_one_hot(labels, 10) 
            optimizer.zero_grad()
            outputs = bp_model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        train_loss = total_loss / num_batches
        val_loss, val_acc = bp_model.evaluate(test_loader)
        bp_train_losses.append(train_loss)
        bp_val_losses.append(val_loss)
        bp_val_accs.append(val_acc)
        print(f"Backprop Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    bp_time = time.time() - bp_start
    if device.type == "cuda":
        bp_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        torch.cuda.reset_peak_memory_stats(device)
    else:
        bp_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()
        tracemalloc.start()

    print()

    torch.manual_seed(42)
    pc_start = time.time()
    pc_model = PredictiveCodingNetwork(layer_ns=[784, 128, 64, 10], lr=0.001, device=device)

    pc_train_energies = []
    pc_val_losses = []
    pc_val_accs = []

    for epoch in range(1, epochs + 1):
        total_energy = 0.0
        num_batches = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            targets = to_one_hot(labels, 10)
            energy_history = pc_model.infer_and_learn(images, targets, n_inference_steps=20, lr_z=0.01)
            total_energy += energy_history[-1]["total_energy"]
            num_batches += 1

        train_energy = total_energy / num_batches
        val_loss, val_acc = pc_model.evaluate(test_loader)
        pc_train_energies.append(train_energy)
        pc_val_losses.append(val_loss)
        pc_val_accs.append(val_acc)
        print(f"PC Epoch {epoch}/{epochs} | Train Energy: {train_energy:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    pc_time = time.time() - pc_start
    if device.type == "cuda":
        pc_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    else:
        pc_mem = tracemalloc.get_traced_memory()[1] / (1024 * 1024)
        tracemalloc.stop()

    print()
    print(f"Backprop: Time = {bp_time:.2f}s | Memory = {bp_mem:.2f} MB | Val Acc = {bp_val_accs[-1]:.2f}%")
    print(f"PC:       Time = {pc_time:.2f}s | Memory = {pc_mem:.2f} MB | Val Acc = {pc_val_accs[-1]:.2f}%")

    epochs_range = list(range(1, epochs + 1))
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, bp_val_accs, 'o-', label=f"Backprop ({bp_val_accs[-1]:.1f}%)")
    plt.plot(epochs_range, pc_val_accs, 's--', label=f"PC ({pc_val_accs[-1]:.1f}%)")
    plt.xlabel("Epoch")
    plt.ylabel("Val Accuracy (%)")
    plt.title("Accuracy Comparison")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, bp_train_losses, 'o-', label="Backprop Loss")
    plt.plot(epochs_range, pc_train_energies, 's--', label="PC Energy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Energy")
    plt.title("Training Loss & Energy")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("comparison_plot.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    run_experiment()
