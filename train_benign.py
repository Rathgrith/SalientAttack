from core.utils.utility import *
from torch.utils.data import Dataset
import torchvision

if __name__ == "__main__":
    schedule = {
        "device": "GPU",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_num": 1,
        "benign_training": True,
        "batch_size": 128,
        "num_workers": 16,
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "gamma": 0.1,
        "schedule": [150, 180],
        "model_type": "resnet50",
        "epochs": 200,
        "log_iteration_interval": 100,
        "test_epoch_interval": 20,
        "save_epoch_interval": 50,
        "save_dir": "benign_models",
        "experiment_name": "train_benign_DatasetFolder-CIFAR10_resnet18",
    }
    # for i in range(5):
    #     dataset = torchvision.datasets.DatasetFolder
    #     trainset, testset = makeDataLoaders(os.path.join("/output/TRAINSUB/","SET"+str(i)) ,"../input0/cifar10/cifar10/test")
    #     make_benign_model(trainset, schedule, testset)

    root_path = "../input0/cifar10/cifar10/"
    trainset_folder = "train"
    dataset = torchvision.datasets.DatasetFolder
    trainset, testset = makeDataLoaders(
        os.path.join(root_path, trainset_folder), "../input0/cifar10/cifar10/test"
    )
    make_benign_model(trainset, schedule, testset)
