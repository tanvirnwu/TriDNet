import Utils

# ===== Inference =====
test_path = r'G:\TripDNet\Data\RH Test 1'
gt_image = r"G:\LightDehazeNet\Data\Train 1\EH\Train\GT\OOTSGT__43.jpg"
hazy_image = r"G:\LightDehazeNet\Data\Train 1\EH\Train\Haze\OOTSEHL9_43.jpg"

d
gt_folder= r'G:\TripDNet\Data\Dehazer Testset\Cloud Test\GT'
hazy_folder = r'G:\TripDNet\Data\Dehazer Testset\Cloud Test\Haze'

# Version 1: Use the predicted class from the Haze Classifier: USe 3 different models
# Version 2: Use the predicted class from the Haze Classifier: Use only 1 model
dehazers =['LD_Net_Cloud', 'LD_Net_EH', 'LD_Net_Fog']
Utils.TTCDehazeNet(version = 2, gt_image = gt_image, hazy_image = hazy_image, dehazer = dehazers)
Utils.batch_dehaze_and_evaluate(dehazers='AllDehazer_LD_40_16_le-4_eph_35', gt_folder = gt_folder, hazy_folder = hazy_folder)


# selected_models = ['DenseNet201', 'ResNet152', 'ConvNextLarge']
# test_path = r'G:\TTCDehazeNet\Data\RH Test 2'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\RH Test 3'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\Benchmarking Datasets\Set 1'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\Benchmarking Datasets\Set 2'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\Benchmarking Datasets\Set 3'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\Benchmarking Datasets\Set 4'
# Utils.multiple_inference(test_path, selected_models)
# test_path = r'G:\TripDNet\Data\Benchmarking Datasets\Set 5'
# Utils.multiple_inference(test_path, selected_models)