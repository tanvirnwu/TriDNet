from PIL import Image

import Utils, EngineClassifier
from EngineClassifier.HazeClassifiers import *
import EngineClassifier, EngineDehazer
import os
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim



def get_class_name_from_index(index):
    classes = ['Cloud', 'EH', 'Fog']
    return classes[index]


def multiple_model_inference(model_name, image_path, multiple_inference = False):

    # Load the model
    model_path = os.path.join(r'G:\TriDNet\Storage\Saved_Models\ES Best Models 2', f'{model_name}.pth')

    if model_name == 'ConvNextLarge':
        model = ConvNextLarge()
    elif model_name == 'ResNet152':
        model = ResNet152()
    elif model_name == 'DenseNet201':
        model = DenseNet201()
    else:
        print(f'Please select a valid model name from the list: {model_name}')

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = val_test_transform(image)
    image = image.unsqueeze(0).to(device)  # Add batch dimension and move to the device

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        _, predicted_idx = torch.max(outputs, 1)
        predicted_probabilities = torch.nn.functional.softmax(outputs, dim=1)

    # Retrieve the highest probability
    predicted_probability, _ = predicted_probabilities.max(1)

    # Use the predicted index to get the class name
    predicted_class_name = get_class_name_from_index(predicted_idx)

    if not multiple_inference:
        print(f"Model: {model_name}   | Predicted Class: {predicted_class_name} | Prediction Probability: {predicted_probability.item()}")

    return predicted_class_name, predicted_probability.item()





def multiple_inference(test_path, model_names):
    # print(f'Inside the multiple_inference function.')
    class_folders = os.listdir(test_path)
    predictions = []
    actual_labels = []

    for class_folder in class_folders:
        class_folder_path = os.path.join(test_path, class_folder)
        if os.path.isdir(class_folder_path):
            for image_file in os.listdir(class_folder_path):
                image_path = os.path.join(class_folder_path, image_file)
                if os.path.isfile(image_path):
                    predicted_class, probability = Utils.multiple_model_inference(model_names[0], image_path, multiple_inference = True)
                    model_1_output = (predicted_class, probability)
                    predicted_class, probability = Utils.multiple_model_inference(model_names[1], image_path, multiple_inference = True)
                    model_2_output = (predicted_class, probability)
                    predicted_class, probability = Utils.multiple_model_inference(model_names[2], image_path, multiple_inference = True)
                    model_3_output = (predicted_class, probability)
                    predicted_class_name = EngineClassifier.select_ensemble_output([model_1_output, model_2_output, model_3_output])

                    # predicted_class_name = EngineClassifier.TriDNetInference(image_path, model_names,
                    #                                                               multiple_inference=True, actual_labels=class_folder)

                    # print(f'Predicted Class: {predicted_class_name} | Actual Class: {class_folder}')
                    predictions.append(predicted_class_name)
                    actual_labels.append(class_folder)

    # Calculate Metrics
    precision = precision_score(actual_labels, predictions, average='macro')
    recall = recall_score(actual_labels, predictions, average='macro')
    f1 = f1_score(actual_labels, predictions, average='macro')
    mcc = matthews_corrcoef(actual_labels, predictions)
    top1_accuracy = sum([1 for i, j in zip(actual_labels, predictions) if i == j]) / len(actual_labels)

    # Extract Test Dataset Name
    test_dataset_name = os.path.basename(os.path.normpath(test_path))

    # File path for CSV
    csv_file_path = r'G:\TriDNet\Storage\Evaluation Metrics\Evaluation_Metrics_final.csv'

    # Check if the file exists
    file_exists = os.path.isfile(csv_file_path)

    # Metrics for this run
    new_metrics = {
        "Test Dataset": test_dataset_name,
        "Top-1 Accuracy": round(top1_accuracy, 2),
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2),
        "MCC": round(mcc, 2)}

    print(f'Testset: {test_dataset_name} | Accuracy: {top1_accuracy} | Precision: {precision} | Recall: {recall} | F1 Score: {f1} | MCC: {mcc}')
    # Convert new_metrics to a DataFrame
    new_metrics_df = pd.DataFrame([new_metrics])

    # Append to CSV file, with headers only if file does not exist
    new_metrics_df.to_csv(csv_file_path, mode='a', index=False, header=not file_exists)

    # Generate and Save Confusion Matrix
    cm = confusion_matrix(actual_labels, predictions, labels = class_folders)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels = class_folders, yticklabels = class_folders)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(r'G:\TriDNet\Storage\Confusion Matrix\\' + test_dataset_name + '_Confusion_Matrix_4.jpg')



# ====== Inference for Dehazer ======

from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import numpy as np
from PIL import Image


def load_model(model_path):
    model = EngineDehazer.LightDehaze_Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def preprocess_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return val_test_transform(image).unsqueeze(0)  # Add batch dimension

# def tensor_to_pil(tensor):
#     transform = transforms.ToPILImage()
#     return transform(tensor.squeeze(0))  # Remove batch dimension

def single_dehaze_inference(dehazer_model_names, gt_image, hazy_image, predicted_class_name):
    gt_image_path = gt_image
    hazy_image_path = hazy_image
    if predicted_class_name == 'Cloud':
        dehazer_model_name = 'CloudDehazer_100_16_le-4_eph_44'
        dehazer_model_path = os.path.join(r"G:\TriDNet\Storage\Saved_Models\Dehazers", f'{dehazer_model_names[0]}.pth')
    elif predicted_class_name == 'EH':
        dehazer_model_name = 'EHDehazer_100_16_le-4_eph_44'
        dehazer_model_path = os.path.join(r"G:\TriDNet\Storage\Saved_Models\Dehazers", f'{dehazer_model_names[1]}.pth')
    elif predicted_class_name == 'Fog':
        dehazer_model_name = 'FogDehazer_100_16_le-4_eph_44'
        dehazer_model_path = os.path.join(r"G:\TriDNet\Storage\Saved_Models\Dehazers", f'{dehazer_model_names[2]}.pth')
    else:
        print(f'Please select a valid model name from the list: {predicted_class_name}')

    # dehazer_model_name = 'GD_Net'
    # dehazer_model_path = os.path.join(r"G:\TriDNet\Storage\Saved_Models\Dehazers", f'{dehazer_model_name}.pth')


    model = load_model(dehazer_model_path)
    # hazy_image = Image.open(hazy_image)
    image_tensor = preprocess_image(hazy_image)

    with torch.no_grad():
        output_tensor = model(image_tensor)

    # Convert tensors to PIL images
    transform = transforms.ToPILImage()
    hazy_image = transform(image_tensor.squeeze(0))
    dehazed_image = transform(output_tensor.squeeze(0))

    if gt_image:
        gt_tensor = preprocess_image(gt_image)
        gt_image = transform(gt_tensor.squeeze(0))
        ncols = 3
    else:
        ncols = 2

    # Visualize the images side by side
    fig, ax = plt.subplots(1, ncols, figsize=(15, 6))
    ax[0].imshow(hazy_image)
    ax[0].set_title('Dehazed Image | Predicted Class: ' + predicted_class_name)
    ax[0].axis('off')

    ax[1].imshow(dehazed_image)
    ax[1].set_title('Dehazed Image')
    ax[1].axis('off')

    if gt_image:
        ax[2].imshow(gt_image)
        ax[2].set_title('Ground Truth Image')
        ax[2].axis('off')

    plt.show()

    # Save the dehazed image
    dehazed_image_filename = os.path.basename(hazy_image_path).split('.')[0] + "_dehazed.jpg"
    dehazed_image_path = os.path.join(r"G:\TriDNet\Storage\Dehazed Images", dehazed_image_filename)
    dehazed_image.save(dehazed_image_path)

    if gt_image:

        psnr,ssim = Utils.calculate_psnr_ssim(gt_image_path, gt_image_path)
        print(f'GT VS GT Image | PSNR: {psnr} | SSIM of : {ssim}')

        psnr, ssim = Utils.calculate_psnr_ssim(gt_image_path, hazy_image_path)
        print(f'GT VS Dehazed Image | PSNR: {psnr} | SSIM of : {ssim}')

        # Calculate PSNR and SSIM
        resize_gt_image = Utils.transform_and_save_image(gt_image_path, r"G:\TriDNet\Storage\Dehazed Images", 512)
        psnr, ssim = Utils.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)
        print(f'GT VS Dehazed Image | PSNR: {psnr} | SSIM of : {ssim}')

    return output_tensor


selected_models = ['DenseNet201', 'ResNet152', 'ConvNextLarge']


# Version 1: Use the predicted class from the Haze Classifier: USe 3 different models
# Version 2: Use the predicted class from the Haze Classifier: Use only 1 model

def TriDNet(version = None, gt_image = None, hazy_image = None, model_names = selected_models, dehazer = None):
    # print(f'Inside the TriDNet function. 1')

    # Version 2: Use the predicted class from the Haze Classifier: Use only 1 model
    if version == 2:
        if os.path.isfile(hazy_image):
            predicted_class,_,_ = Utils.single_model_classification_inference(hazy_image)
            single_dehaze_inference(dehazer, gt_image, hazy_image, predicted_class_name=predicted_class)
        else:
            print('Version 2 can only inference on Single Image. Please provide a single image path.')

    # Version 1: Use the predicted class from the Haze Classifier: USe 3 different models
    elif version == 1:
        if os.path.isfile(hazy_image):
            final_class = EngineClassifier.TriDNetV1Inference(hazy_image, model_names)
            print(f'Final Class: {final_class}')
            single_dehaze_inference(dehazer, gt_image, hazy_image, predicted_class_name=final_class)
        elif os.path.isdir(hazy_image):
            multiple_inference(hazy_image, model_names)

    elif version == None:
        print('Please select a version number: "1" or "2".')

    else:
        print('Please select a valid version number: "1" or "2".')




