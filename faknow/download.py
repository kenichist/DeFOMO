import kagglehub

# Download latest version
path = kagglehub.dataset_download("doanquanvietnamca/liar-dataset")

print("Path to dataset files:", path)