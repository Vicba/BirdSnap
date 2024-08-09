import argparse
import torch
import os
import random
from torchvision import transforms
from sklearn.model_selection import train_test_split
from data_setup import BirdDataset, create_dataloaders, update_data
from engine import train
from model_builder import get_model
import shutil


def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Switching to CPU")
        args.device = "cpu"

    print('Starting training...')
    bucket_name = os.getenv('STORAGE_BUCKET')

    if args.refresh_data:
        update_data(data_dir=args.data_dir, classes=args.classes, bucket_name=bucket_name)    
        print("Data setup completed successfully")
        shutil.rmtree(args.data_dir)

    print("Data setup completed successfully")

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = BirdDataset(bucket_name=bucket_name, transform=transform)

    rand_10 = random.sample(range(0, len(dataset)), 10)
    for i in rand_10:
        image, label = dataset[i]
        print(f"Image shape: {image.size()}, {image.shape}")
        print(f"Label: {label}")
        print(f"Label in class name: {dataset.get_idx_to_class()[label]}")

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=args.seed)
    print(f"Train data: {len(train_data)}")
    print(f"Test data: {len(test_data)}")

    train_dataloader, test_dataloader = create_dataloaders(train_data, test_data, args.batch_size)
    train_features_batch, train_labels_batch = next(iter(train_dataloader))
    print(f"Feature batch shape: {train_features_batch.size()}, {train_features_batch.shape}")
    print(f"Labels batch shape: {train_labels_batch.size()}, {train_labels_batch.shape}")
    print(f"Number of batches: {len(train_dataloader)}")

    model = get_model(num_classes=args.classes).to(args.device)
        
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model = train(model=model,
                wandb_project=args.wandb_project,
                wandb_model_name=args.model_name,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                epochs=args.epochs,
                save_frequency=args.save_frequency,
                device=args.device)  




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='data directory')
    parser.add_argument('--classes', type=int, default=20, help='number of classes')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--save-frequency', type=int, default=2, help='save frequency')
    parser.add_argument('--wandb_project', type=str, default='bird-classification', help='wandb project name')
    parser.add_argument('--model_name', type=str, default='bird-classifier', help='wandb model name')
    parser.add_argument("--refresh_data", action='store_true', help="If set, refresh the data")
    parser.add_argument("--device", type=str, default='cuda', help="Device to train the model")
    parser.add_argument('--seed', type=int, default=42, help='random seed')

    main(parser.parse_args())


