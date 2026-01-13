from EngineClassifier import *
import os
import Utils
from PIL import Image

def select_ensemble_output(model_outputs):
    """
    Selects the ensemble output based on the given model outputs.
    Prioritizes Model 1's output in case of a tie with equal probabilities.

    Parameters:
    model_outputs (list of tuples): Each tuple contains (class_output, probability) for each model.
                                    Assumes model_outputs[0] is from Model 1.

    Returns:
    int: The selected class output.
    """
    # Check if two or more models agree on the same class
    class_counts = {}
    for output, _ in model_outputs:
        if output in class_counts:
            class_counts[output] += 1
        else:
            class_counts[output] = 1

    for class_output, count in class_counts.items():
        if count >= 2:
            return class_output

    # If all models disagree with equal probabilities, prioritize Model 1's output
    if len(set([prob for _, prob in model_outputs])) == 1:
        return model_outputs[0][0]  # Output from Model 1

    # Otherwise, select the class with the highest probability
    return max(model_outputs, key=lambda x: x[1])[0]


# def select_ensemble_output_1(model_outputs, actual_class):
#     """
#     Selects the ensemble output based on the given model outputs.
#     Prioritizes Model 1's output in case of a tie with equal probabilities.
#     Prioritizes Model 2's output if the actual class is 'Fog'.
#
#     Parameters:
#     model_outputs (list of tuples): Each tuple contains (class_output, probability) for each model.
#                                     Assumes model_outputs[0] is from Model 1.
#     actual_class (str): The actual class of the input.
#
#     Returns:
#     int: The selected class output.
#     """
#     # Prioritize Model 2's output if the actual class is 'Fog'
#     if actual_class == 'Fog':
#         return model_outputs[1][0]  # Output from Model 2
#
#     # Check if two or more models agree on the same class
#     class_counts = {}
#     for output, _ in model_outputs:
#         if output in class_counts:
#             class_counts[output] += 1
#         else:
#             class_counts[output] = 1
#
#     for class_output, count in class_counts.items():
#         if count >= 2:
#             return class_output
#
#     # If all models disagree with equal probabilities, prioritize Model 1's output
#     if len(set([prob for _, prob in model_outputs])) == 1:
#         return model_outputs[0][0]  # Output from Model 1
#
#     # Otherwise, select the class with the highest probability
#     return max(model_outputs, key=lambda x: x[1])[0]



def image_processing(image_path):
    image = Image.open(image_path).convert('RGB')
    image = Utils.val_test_transform(image)
    image = image.unsqueeze(0).to(device)
    return image


# Main function to perform ensemble inference
def TriDNetV1Inference(image_path, model_names, multiple_inference = False):

    predicted_class, probability = Utils.multiple_model_inference(model_names[0], image_path, multiple_inference = multiple_inference)
    model_1_output = (predicted_class, probability)
    predicted_class, probability = Utils.multiple_model_inference(model_names[1], image_path, multiple_inference = multiple_inference)
    model_2_output = (predicted_class, probability)
    predicted_class, probability = Utils.multiple_model_inference(model_names[2], image_path, multiple_inference = multiple_inference)
    model_3_output = (predicted_class, probability)


    final_class = select_ensemble_output([model_1_output, model_2_output, model_3_output])


    return final_class




