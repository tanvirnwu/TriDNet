import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from PIL import Image
import os
import Utils
import torch

def transform_and_save_image(gt_image_path, storage_folder, img_size):

    # Define the transformations
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    # Load the image as a PIL image
    pil_image = Image.open(gt_image_path)

    # Apply the transformations
    transformed_tensor = transform(pil_image)

    # Convert back to PIL image
    transformed_image = transforms.ToPILImage()(transformed_tensor)

    # Save the transformed image
    file_name = os.path.basename(gt_image_path)
    save_path = os.path.join(storage_folder, file_name)
    transformed_image.save(save_path, format='JPEG')
    return save_path


def calculate_psnr(imageA, imageB):
    mse = np.mean((imageA - imageB) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def calculate_psnr_ssim(gt_image_path, hazy_image_path):
    # Read the images
    imageA = cv2.imread(gt_image_path)
    imageB = cv2.imread(hazy_image_path)

    # Convert the images to grayscale
    imageA_gray = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    imageB_gray = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Calculate PSNR and SSIM
    psnr_value = calculate_psnr(imageA_gray, imageB_gray)
    ssim_value = ssim(imageA_gray, imageB_gray)

    return psnr_value, ssim_value


import os

import numpy as np

def calculate_mse(gt_image_path, dehazed_image_path):
    # Load images
    gt_image = Image.open(gt_image_path).convert('RGB')
    dehazed_image = Image.open(dehazed_image_path).convert('RGB')

    # Convert images to numpy arrays
    gt_array = np.array(gt_image)
    dehazed_array = np.array(dehazed_image)

    # Calculate MSE
    mse = np.mean((gt_array - dehazed_array) ** 2)
    return mse


def batch_dehaze_and_evaluate(dehazers:str, gt_folder, hazy_folder):

    dehazer_model_path = os.path.join(r"G:\TriDNet\Storage\Saved_Models", f'{dehazers}.pth')
    model = Utils.load_model(dehazer_model_path)

    psnr_values = []
    ssim_values = []
    mse_values = []

    for hazy_image_filename in os.listdir(hazy_folder):
        hazy_image_path = os.path.join(hazy_folder, hazy_image_filename)
        gt_image_path = os.path.join(gt_folder, hazy_image_filename)  # Assuming GT and hazy images have same filenames

        # Dehaze the image
        hazy_image_tensor = Utils.preprocess_image(hazy_image_path)
        with torch.no_grad():
            dehazed_image_tensor = model(hazy_image_tensor)

        # Convert tensors to PIL images for evaluation
        transform = transforms.ToPILImage()
        dehazed_image = transform(dehazed_image_tensor.squeeze(0))
        dehazed_image_path = os.path.join(r"G:\TriDNet\Storage\Batch Dehazed Images\Dehazed", hazy_image_filename)
        dehazed_image.save(dehazed_image_path)

        print(f'original gt image: {gt_image_path}')
        # Calculate PSNR and SSIM
        resize_gt_image = Utils.transform_and_save_image(gt_image_path,
                                                         r"G:\TriDNet\Storage\Batch Dehazed Images\GT", 512)

        print(f'GT IMage path: {resize_gt_image}')
        print(f'Dehazed Image path: {dehazed_image_path}')

        mse = calculate_mse(resize_gt_image, dehazed_image_path)
        psnr, ssim = Utils.calculate_psnr_ssim(resize_gt_image, dehazed_image_path)

        psnr_values.append(psnr)
        ssim_values.append(ssim)
        mse_values.append(mse)

        print(f"PSNR: {psnr} | SSIM: {ssim} | MSE: {mse}")


    # Calculate average PSNR and SSIM
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_mse = np.mean(mse_values)

    print(f"Average PSNR: {avg_psnr} | Average SSIM: {avg_ssim} | Average MSE: {avg_mse}")

    return avg_psnr, avg_ssim, avg_mse
