from flask import Flask, request, jsonify, render_template
import os
import torch
from torchvision import models, transforms
from PIL import Image
import re
from google.cloud import storage
from io import BytesIO
from utils import bird_names
from dotenv import load_dotenv
import subprocess

load_dotenv()

app = Flask(__name__)

num_classes = 20
bucket_name = os.getenv("STORAGE_BUCKET")
project_id = os.getenv("GCP_PROJECT_ID")
model_dir = 'models'

def create_model():
    """Create a model with the same architecture and output layer as the trained model"""
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model

def get_model():
    """Get the last .pth model checkpoint from Google Cloud Storage"""
    client = storage.Client(project=project_id)
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=model_dir)
    model_files = [blob.name for blob in blobs if blob.name.endswith('.pth')]
    
    if not model_files:
        return None, None

    # Sort the model files numerically by epoch number
    model_files.sort(key=lambda f: int(re.search(r'(\d+)', f).group()))

    latest_model_path = model_files[-1]
    
    # Download the latest model
    blob = bucket.blob(latest_model_path)
    model_data = BytesIO()
    blob.download_to_file(model_data)
    model_data.seek(0)
    
    model = create_model()
    model.load_state_dict(torch.load(model_data, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image):
    """Preprocess the uploaded image"""
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image).convert('RGB')
    image = preprocess(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image

def predict(image, model):
    """Predict bird class using the model"""
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)

    probabilities = probabilities.squeeze().tolist()
    sorted_probabilities = sorted(enumerate(probabilities), key=lambda x: x[1], reverse=True)

    return predicted.item(), sorted_probabilities

@app.route('/health')
def health_check():
    return "Everything is fine!"

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        model = get_model()
        if model:
            prediction, probs = predict(file, model)
            response = {
                'predicted_bird_type': bird_names[prediction],
                'probabilities': {bird_names[i]: prob for i, prob in probs[:5]},
            }
            return jsonify(response)
        else:
            return jsonify({'error': 'Model not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0')
