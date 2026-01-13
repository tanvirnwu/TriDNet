import torch, os
from torchvision import transforms


# model_name = 'AlexNet'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_shape = 3
hidden_unit = 32
output_shape = 3
train_size = 0.85
img_size = 512
# num_workers = os.cpu_count()
# print(f"Number of CPU cores available: {num_workers}")

# ========== MODEL PARAMETERS ==========
num_epochs = 40
batch_size = 32
learning_rate = 0.001
scheduler_activate = False
lr_decay_steps = 5
gamma = 0.5
gamma_value_increase_rate = None
stop_gamma_value_increase_epoch = None

# ========== REGULARIZATION PARAMETERS ==========
dropout_rate = None
l2_lambda = 0.0002

# ========= DATA TRANSFORMS =======
# Data transforms for training data
train_transforms = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    # transforms.RandomResizedCrop(299),
    # transforms.RandomCrop(256),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

val_test_transform = transforms.Compose([
                      transforms.Resize((img_size, img_size)),
                      transforms.ToTensor(),
                      # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                      #    std=[0.229, 0.224, 0.225])
     ])


# ========== DATA PATHS ==========
# train_path = r'G:\HazeClassification\Data\Train'
# test_path = r'G:\HazeClassification\Data\Test'
test_path_for_class_name = r'G:\HazeClassification\Data\RH Test 5'

config_file_path = r'G:\HazeClassification\Utils\Config.py'
# Path where you want to save the new text file
# save_file_path = r'G:\HazeClassification\Storage\Save_Configs\\' + model_name + '_Configs.txt'

# ========= STORAGE PATHS =========
evaluation_metrics_csv_dir = r'G:\HazeClassification\Storage\Evaluation Metrics (on RHset)'
auc_pr_save_testset = r"G:\HazeClassification\Storage\AUC_PR (on Testset)"
auc_pr_save_RHset = r"G:\HazeClassification\Storage\AUC_PR (on RHset)"
evaluation_metrics_testset = r'G:\HazeClassification\Storage\Evaluation_Metrics (on Testset)'
evaluation_metrics_RHset = r'G:\HazeClassification\Storage\Evaluation Metrics (on RHset)'
confusion_matrix_save_testset = r'G:\HazeClassification\Storage\Confusion_Matrix (on Testset)'
confusion_matrix_save_RHset = r'G:\HazeClassification\Storage\Confusion_Matrix (on RHset)'
# model_trianing_results_saving_path = r'G:\HazeClassification\Storage\Loss_Curve_Results\\' + model_name + '.pkl'
Loss_Curves = r'G:\HazeClassification\Storage\Loss_Curves'
# model_weights_saving_path = r'G:\HazeClassification\Storage\Saved_Models\\' + model_name + '.pth'

