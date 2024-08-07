import os
import json
import kaggle
import argparse
import shutil
from google.cloud import storage
from collections import Counter
import pandas as pd
import glob
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
import random
from dotenv import load_dotenv
from io import BytesIO

load_dotenv()

def upload_to_gcs(bucket_name, source_folder, destination_folder=""):
    """
    Uploads files to Google Cloud Storage while preserving the directory structure.
    
    Args:
        bucket_name (str): Name of the Google Cloud Storage bucket.
        source_folder (str): Local folder containing files to upload.
        destination_folder (str): Destination folder in the bucket. Default is the root of the bucket.
    """
    storage_client = storage.Client(project=os.getenv('GCP_PROJECT_ID'))
    bucket = storage_client.bucket(bucket_name)

    # remove everything in bucket
    blobs = bucket.list_blobs()
    bucket.delete_blobs(blobs)

    for root, _, files in os.walk(source_folder):
        for file in files:
            local_path = os.path.join(root, file).replace("\\", "/")
            relative_path = os.path.relpath(local_path, source_folder).replace("\\", "/")
            blob_name = os.path.join(destination_folder, relative_path).replace("\\", "/")

            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_path)

def download_kaggle_dataset(kaggle_dataset, data_dir="data"):
    """ Downloads a dataset from Kaggle and uploads it to Google Cloud Storage. 
    Args:
        kaggle_dataset: Kaggle dataset name in the format <name>/<dataset-name>.
        bucket_name: Name of the Google Cloud Storage bucket to upload the data.
    """
    # Retrieve Kaggle credentials from environment variables
    kaggle_username = os.getenv('KAGGLE_USERNAME')
    kaggle_key = os.getenv('KAGGLE_KEY')

    if os.path.exists(data_dir):
        shutil.rmtree(data_dir)

    if kaggle_username is None or kaggle_key is None:
        raise ValueError("Kaggle credentials not found in environment variables.")

    # Assuming raw_data_path is always in the format <name>/<dataset-name>
    parts = kaggle_dataset.split("/")
    if len(parts) != 2:
        raise ValueError("Invalid data path format. Expected <name>/<dataset-name>.")

    # Initialize Kaggle API
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
        f.write(json.dumps({"username": kaggle_username, "key": kaggle_key}))

    kaggle.api.authenticate()

    # Download dataset from Kaggle
    print(f"Downloading dataset {kaggle_dataset} from Kaggle")
    kaggle.api.dataset_download_files(dataset=kaggle_dataset, path=data_dir, unzip=True, quiet=False)


def get_top_bird_species(data_dir, num_classes):
    """Get the top 20 bird species by image count.
    Args:
        data_dir: Directory containing bird images organized in subdirectories by species.
    Returns:
        A list of the top 20 bird species.
    """
    all_images = glob.glob(os.path.join(data_dir, '**', '*.jpg'), recursive=True)
    species = [os.path.basename(os.path.dirname(img)) for img in all_images]
    species_counts = Counter(species)
    top_species = [species for species, _ in species_counts.most_common(num_classes)]
    return top_species


def filter_top_species(data_dir, top_species):
    """Filter the data to only keep the top 20 bird species.
    Args:
        data_dir: Directory containing bird images organized in subdirectories by species.
        top_20_species: List of the top 20 bird species.
    Returns:
        Path to the directory containing only the top 20 bird species.
    """
    if os.path.exists(os.path.join(data_dir, "valid")):
        shutil.rmtree(os.path.join(data_dir, "valid"))
    if os.path.exists(os.path.join(data_dir, "test")):
        shutil.rmtree(os.path.join(data_dir, "test"))

    for root, dirs, files in os.walk(data_dir, topdown=False):
        for file in files:
            species = os.path.basename(root)
            if species not in top_species:
                os.remove(os.path.join(root, file))

        if not os.listdir(root):
            os.rmdir(root)

class BirdDataset(Dataset):
    def __init__(self, bucket_name, transform=None):
        self.bucket_name = bucket_name
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        self.client = storage.Client()
        self.bucket = self.client.bucket(self.bucket_name)
        
        self._load_data()

    def _load_data(self):
        blobs = list(self.client.list_blobs(self.bucket))

        for blob in blobs:
            # Skip directories or non-image files if needed
            if not blob.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            class_name = os.path.dirname(blob.name).strip('/')
            img_path = blob.name

            if class_name not in self.class_to_idx:
                self.class_to_idx[class_name] = len(self.class_to_idx)
            class_idx = self.class_to_idx[class_name]

            self.image_paths.append(img_path)
            self.labels.append(class_idx)

        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # print(img_path)
        label = self.labels[idx]

        blob = self.bucket.blob(img_path)
        img_bytes = blob.download_as_bytes()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_idx_to_class(self):
        return self.idx_to_class

# class BirdDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_paths = []
#         self.labels = []
#         self.class_to_idx = {}

#         # for idx, class_name in enumerate(os.listdir(root_dir)):
#         #     class_path = os.path.join(root_dir, class_name)
#         #     if os.path.isdir(class_path):
#         #         self.class_to_idx[class_name] = idx
#         #         for img_name in os.listdir(class_path):
#         #             img_path = os.path.join(class_path, img_name)
#         #             self.image_paths.append(img_path)
#         #             self.labels.append(idx)

#         # root dir is gcp bucket
#         if root_dir.startswith("gs://"):
#             storage_client = storage.Client(project=os.getenv('GCP_PROJECT_ID'))
#             bucket_name = root_dir.split("/")[2]
#             bucket = storage_client.bucket(bucket_name)

#             blobs = bucket.list_blobs()
#             for blob in blobs:
#                 blob_name = blob.name
#                 class_name = blob_name.split("/")[0]
#                 if class_name not in self.class_to_idx:
#                     self.class_to_idx[class_name] = len(self.class_to_idx)
#                 self.image_paths.append(f"gs://{bucket_name}/{blob_name}")
#                 self.labels.append(self.class_to_idx[class_name])


#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         img_path = self.image_paths[idx].replace("\\", "/")
#         label = self.labels[idx]
#         # print(img_path)
#         image = Image.open(img_path).convert("RGB")

#         if self.transform:
#             image = self.transform(image)

#         return image, label
    
#     def get_idx_to_class(self):
#         return {v: k for k, v in self.class_to_idx.items()}
    

def create_dataloaders(
    train_data: BirdDataset,
    test_data: BirdDataset,
    batch_size: int, 
    num_workers: int=os.cpu_count(),
):
    """Creates training and testing DataLoaders.

    Args:
      train_data: Training dataset.
      test_data: Testing dataset.
      batch_size: Number of samples per batch in each of the DataLoaders.
      num_workers: An integer for number of workers per DataLoader.

    Returns:
      A tuple of (train_dataloader, test_dataloader, class_names).
      Where class_names is a list of the target classes.
    """

    train_dataloader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_dataloader, test_dataloader

def update_data(data_dir="data", classes=20, bucket_name=None):
    download_kaggle_dataset("gpiosenka/100-bird-species")
    print("Data downloaded successfully")
    
    top_species = get_top_bird_species(data_dir, classes)
    print("Top species by image count:", top_species)

    filter_top_species(data_dir, top_species)

    # copy everything from train folder to parent folder data
    train_data_path = data_dir + "/train"
    for file in os.listdir(train_data_path):
        shutil.move(train_data_path + "/" + file, data_dir)
    os.rmdir(train_data_path)

    if bucket_name:
        upload_to_gcs(bucket_name=bucket_name, source_folder=data_dir)
        print("Filtered data uploaded to GCS successfully")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--classes', type=int, default=20, help='number of classes')
    parser.add_argument('--model_dir', type=str, default='model', help='model directory')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument("--refresh_data", action='store_true', help="If set, refresh the data")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    bucket_name = os.getenv('STORAGE_BUCKET')

    if args.refresh_data:
        update_data(data_dir=args.data_dir, classes=args.classes, bucket_name=bucket_name)    
        print("Data setup completed successfully")

    print("Data setup completed successfully")

    # Transformations
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Create Dataset
    dataset = BirdDataset(root_dir=args.data_dir, transform=transform)

    # get first 5 items from dataset
    print(len(dataset))
    # get 10 random numbers
    rand_10 = random.sample(range(0, len(dataset)), 10)
    for i in rand_10:
        image, label = dataset[i]
        print(f"Image shape: {image.size()}, {image.shape}")
        print(f"Label: {label}")
        print(f"Label in class name: {dataset.get_idx_to_class()[label]}")

    # train test split
    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=args.seed)

    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    # Create DataLoaders
    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, args.batch_size)
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features_batch.size()}, {train_features_batch.shape}")
    print(f"Labels batch shape: {train_labels_batch.size()}, {train_labels_batch.shape}")
    print(f"Number of batches: {len(train_dataloader)}")

    


