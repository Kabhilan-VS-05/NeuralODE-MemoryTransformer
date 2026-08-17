import zipfile
import os

def zipdir(path, ziph):
    for root, dirs, files in os.walk(path):
        dirs[:] = [d for d in dirs if d not in ['.venv', 'venv', 'env', '.git', 'cached_features', 'cifar100_data', 'Zip files', '__pycache__', '.vscode']]
        for file in files:
            if file.endswith('.pt') or file.endswith('.zip') or file.endswith('.tar.gz'):
                continue
            file_path = os.path.join(root, file)
            ziph.write(file_path, os.path.relpath(file_path, path))

os.makedirs('Zip files', exist_ok=True)
with zipfile.ZipFile('Zip files/backup_v4.1.zip', 'w', zipfile.ZIP_DEFLATED) as zipf:
    zipdir('.', zipf)
print("Successfully created Zip files/backup_v4.1.zip")
