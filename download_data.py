from src.shared.data_loader import get_data_loader

print("Downloading MNIST data safely...")
loader = get_data_loader()
print("Download complete! Now you can run mpiexec.")