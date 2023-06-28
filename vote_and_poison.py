from core.utils.utility import *
from torch.utils.data import Dataset
import torchvision
import os
import tqdm
import torch
import random
import numpy as np

if __name__ == "__main__":
    # Define Configurations
    # setting seed
    seed = 0
    """
        Available targets for CIFAR-10
        'airplane', 'automobile', 'bird', 0-2
        'cat', 'deer', 'dog', 3-5
        'frog', 'horse', 'ship', 'truck' 5-9
    """
    target = 9
    # setting dataset path
    root_path = "../input0/cifar10/cifar10/"
    set_folder = "train"
    """
        Attacks avalilable: "BadNets", "Blended"
        
    """
    attack_method, trigger_weight = "Blended", 0.15
    # attack_method, trigger_weight = "BadNets", 1.0
    # BadNets 1.0, Blended 0.15
    voting_models = [
        "./benign_models/vgg16bn_experiment001/ckpt_epoch_200.pth",
        "./benign_models/vgg16bn_experiment002/ckpt_epoch_200.pth",
        "./benign_models/vgg16bn_experiment003/ckpt_epoch_200.pth",
        "./benign_models/vgg16bn_experiment004/ckpt_epoch_200.pth",
        "./benign_models/vgg16bn_experiment005/ckpt_epoch_200.pth",
    ]
    # voting_models =["./benign_models/resnet18_experiment001/ckpt_epoch_200.pth",
    #         "./benign_models/resnet18_experiment002/ckpt_epoch_200.pth",
    #         "./benign_models/resnet18_experiment003/ckpt_epoch_200.pth",
    #         "./benign_models/resnet18_experiment004/ckpt_epoch_200.pth",
    #         "./benign_models/resnet18_experiment005/ckpt_epoch_200.pth"]
    poison_rate = 0.01
    num_class = 10
    kmeans_k = 25
    isSalient = True
    save_dir = "attack_experiments"
    experiment_name = "Salient_Blended_CIFAR10_five_folds_vgg16BN_voting_attacking_VGG16BN_for_reproduce"

    # Execution
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    path = os.path.join(root_path, set_folder)
    freeze_dataset_JSON(path)
    # won't make difference for following attacking
    # just a placeholder for init models here...
    default_parameters = make_parameters(attack="Blended", weight_parameter=0.15)
    poisoned_label = sorted(os.listdir(path))[target]
    print("Sample is poisoned with the target:", poisoned_label)

    schedule = {
        "device": "GPU",
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_num": 1,
        "benign_training": True,
        "batch_size": 1000,
        "num_workers": 24,
        "schedule": [30, 50],
        "test_model": "",
        "save_dir": "vote_test",
        "experiment_name": "test",
    }

    test_set_list = sorted(os.listdir(os.path.join("./TRAINSUB")))

    for i in range(len(test_set_list)):
        test_set_list[i] = os.path.join("./TRAINSUB", test_set_list[i])

    TargetNet = core.Blended(
        train_dataset=default_parameters["trainset"],
        test_dataset=default_parameters["testset"],
        model=core.models.vgg16_bn(),
        # model=core.models.BaselineMNISTNetwork(),
        poisoned_rate=0.01,
        pattern=default_parameters["pattern"],
        weight=default_parameters["weight"],
        loss=nn.CrossEntropyLoss(),
        y_target=1,
        # poisoned_transform_index=0,
        poisoned_target_transform_index=0,
        schedule=None,
    )

    image_title_dict = freeze_dataset_JSON(path)
    possibility_lists = []

    possibility_lists_dict = {}
    model_group_name = "possibility_" + voting_models[0].split("/")[2][:-3]
    if model_group_name not in os.listdir():
        os.makedirs(model_group_name)

    print("Start benign model inference")
    for idx, model in tqdm.tqdm(
        enumerate(voting_models),
        desc="  Benign models infering",
        total=len(voting_models),
    ):
        schedule["test_model"] = model
        possibility_list = TargetNet.infer(
            schedule,
            image_title_dict=image_title_dict,
            test_datasets=test_set_list,
            train_index=idx,
        )
        # print(possibility_list)
        possibility_lists.append(possibility_list)

    possibility_lists_dict["possibility_lists_dict"] = possibility_lists
    local_possibility_path = os.path.join(
        model_group_name, "possibility_lists_dict.json"
    )
    json_write_file(local_possibility_path, possibility_lists_dict)
    possibility_lists_dict = json_read_file(local_possibility_path)
    # print(possibility_lists_dict)
    possibility_lists = possibility_lists_dict["possibility_lists_dict"]
    borda_res = borda_count(possibility_lists)
    index_to_path_dict = json_read_file("idx_to_path.json")

    vote_path, vote_score = get_borda_path(
        borda_res,
        poison_rate,
        index_to_path_dict,
        isSalient=isSalient,
        classes=num_class,
        poisoned_label=poisoned_label,
        kmeans_k=kmeans_k,
    )

    for idx in range(len(vote_path)):
        vote_path[idx] = os.path.join("../input0/cifar10/cifar10/train", vote_path[idx])
    visualize_image(vote_path, vote_score, len(vote_path))
    vote_dict = {}
    vote_dict["vote_result"] = vote_path
    vote_res_path = os.path.join(model_group_name, "vote_result/experiment01.json")
    json_write_file("vote_result/experiment01.json", vote_dict)

    attack_parameters = make_parameters(attack=attack_method, weight_parameter=trigger_weight)
    # attack_parameters = make_parameters(attack="BadNets",weight_parameter=1.0)
    schedule = {
        "seed": seed,
        "target_label": poisoned_label,
        "attack_method": attack_method,
        "device": "GPU",
        "Poison_ratio": poison_rate,
        "num_classes": num_class,
        "k_value": kmeans_k,
        "Salient_selection": isSalient,
        "attack_method": attack_method,
        "CUDA_VISIBLE_DEVICES": "0",
        "GPU_num": 1,
        "benign_training": False,
        "batch_size": 512,
        "num_workers": 24,
        "lr": 0.01,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "gamma": 0.1,
        "schedule": [30, 50, 60],
        "model_type": "VGG16BN",
        "epochs": 70,
        "log_iteration_interval": 45,
        "test_epoch_interval": 2,
        "save_epoch_interval": 70,
        "save_dir": save_dir,
        "experiment_name": experiment_name,
    }
    
    TargetNet = core.BadNets(
        train_dataset=attack_parameters["trainset"],
        test_dataset=attack_parameters["testset"],
        model=core.models.vgg16_bn(),
        # model=core.models.BaselineMNISTNetwork(),
        poisoned_rate=poison_rate,
        pattern=attack_parameters["pattern"],
        weight=attack_parameters["weight"],
        loss=nn.CrossEntropyLoss(),
        y_target=target,
        # poisoned_transform_index=0,
        specific_path=vote_path,
        poisoned_target_transform_index=0,
        schedule=None,
    )
    TargetNet.train(schedule)
