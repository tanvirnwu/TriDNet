import Utils, EngineClassifier
import os
from PIL import Image
import torch
from torchvision import transforms
from Utils.Config import *

def get_class_name_from_index(index, test_path):
    # Ensure the classes are always listed in the same order by sorting
    classes = sorted(os.listdir(test_path))
    # Return the class name that corresponds to the index
    return classes[index]


def single_model_classification_inference(image_path, test_path = Utils.test_path_for_class_name, transform = Utils.val_test_transform):
    # Load the model
    model_name = 'ResNet152'
    model_path = os.path.join(r'G:\TripDNet\Storage\Saved_Models', f'{model_name}.pth')
    model = EngineClassifier.ResNet152()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Load and transform the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to the device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Retrieve the highest probability
    predicted_probability, _ = predicted_probabilities.max(1)

    # Extract the actual class name from the image path
    actual_class_name = os.path.basename(os.path.dirname(image_path))

    # Use the predicted index to get the class name
    predicted_class_name = get_class_name_from_index(predicted_idx, test_path)

    # Print the results
    print(f"Actual class: {actual_class_name}")
    print(f"Predicted class: {predicted_class_name}")
    print(f"Predicted probability: {predicted_probability.item()}")

    return predicted_class_name, actual_class_name, predicted_probability.item()