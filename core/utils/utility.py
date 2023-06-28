import base64
import os
import core
import json
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
from torchvision.transforms import (
    Compose,
    ToTensor,
    PILToTensor,
    RandomHorizontalFlip,
    Resize,
)
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import math
from shutil import copyfile
import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# import torchvision.models as models
from torch.utils.data import DataLoader
from imutils import build_montages
import shutil

# /openbayes/home/TRAINSUB


def json_write_file(path, content):
    with open(path, "w") as f:
        json.dump(content, f)


def json_read_file(path, parameter):
    with open(path, "r") as fp:
        parameter = json.load(fp)
    return parameter


def get_benign_model(trainsets, schedule, testset):
    for trainset in trainsets:
        make_benign_model(trainset, schedule, testset)


def make_benign_model(trainset, schedule, testset):
    # save_dir = "vote_train" if IS_ON_SAME_TRAIN_SET else "vote_test"
    pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
    pattern[0, -3:, -3:] = 255
    weight = torch.zeros((1, 32, 32), dtype=torch.float32)
    weight[0, -3:, -3:] = 1.0

    if schedule["benign_training"] != True:
        print("not benign setting! skip training...")
        return
    model_name = schedule["model_type"]
    if model_name == "resnet18":
        TargetNet = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(18),
            # model=core.models.BaselineMNISTNetwork(),
            poisoned_rate=0.01,
            pattern=pattern,
            weight=weight,
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            # poisoned_transform_index=0,
            poisoned_target_transform_index=0,
            schedule=None,
        )

    elif model_name == "resnet50":
        TargetNet = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.ResNet(50),
            # model=core.models.BaselineMNISTNetwork(),
            poisoned_rate=0.01,
            pattern=pattern,
            weight=weight,
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            # poisoned_transform_index=0,
            poisoned_target_transform_index=0,
            schedule=None,
        )

    elif model_name == "vgg16bn":
        TargetNet = core.BadNets(
            train_dataset=trainset,
            test_dataset=testset,
            model=core.models.vgg16_bn(),
            # model=core.models.BaselineMNISTNetwork(),
            poisoned_rate=0.01,
            pattern=pattern,
            weight=weight,
            loss=nn.CrossEntropyLoss(),
            y_target=1,
            # poisoned_transform_index=0,
            poisoned_target_transform_index=0,
            schedule=None,
        )
    TargetNet.train(schedule)


def split_dataset(source_train_dataset_path, fold, dataset_name, target_folder):
    if dataset_name == "cifar10":
        train_dataset_path = source_train_dataset_path
        classes = os.listdir(train_dataset_path)
        dataset_counts = fold
        random_list = [np.random.permutation(np.arange(5000)) for x in range(10)]
        os.makedirs(target_folder)
        sub_train_set = "SET"
        root_path = os.path.join(target_folder, sub_train_set)
        for i in range(dataset_counts):
            sub_dataset_path = root_path + str(i)
            os.makedirs(sub_dataset_path)
            for single_class in classes:
                os.makedirs(os.path.join(sub_dataset_path, single_class))
        count = 0
        for single_class in classes:
            class_path = os.path.join(train_dataset_path, single_class)
            for i in range(dataset_counts):
                destination_dir_path = os.path.join((root_path + str(i)), single_class)
                images = os.listdir(class_path)
                chosen_images = np.array(images)[
                    random_list[count][1000 * i : 1000 * i + 1000]
                ]
                for chosen_image in chosen_images:
                    source_file_path = os.path.join(class_path, chosen_image)
                    destination_path = os.path.join(destination_dir_path, chosen_image)
                    copyfile(source_file_path, destination_path)
            count += 1


def make_parameters(attack="BadNets", weight_parameter=1.0):
    # configure Attacks patterns
    if attack == "BadNets":
        pattern = torch.zeros((1, 32, 32), dtype=torch.uint8)
        pattern[0, -3:, -3:] = 255
        weight = torch.zeros((1, 32, 32), dtype=torch.float32)
        weight[0, -3:, -3:] = weight_parameter
    elif attack == "Blended":
        trigger = cv2.imread("./trigger/0.jpg", cv2.IMREAD_COLOR)
        trigger = cv2.resize(trigger, (32, 32)).astype(np.int32)
        transform = transforms.ToTensor()
        pattern = transform(trigger)
        weight = torch.zeros((1, 32, 32), dtype=torch.float32)
        weight[:, :, :] = weight_parameter
        # print(pattern)

    # configure Dataloaders
    transform_train = Compose([ToTensor(), RandomHorizontalFlip()])
    dataset = torchvision.datasets.DatasetFolder
    trainset = dataset(
        root="../input0/cifar10/cifar10/train",
        loader=cv2.imread,
        extensions=("png",),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None,
    )

    transform_test = Compose([ToTensor()])
    testset = dataset(
        root="../input0/cifar10/cifar10/test",
        loader=cv2.imread,
        extensions=("png",),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None,
    )
    res = {}
    res["pattern"] = pattern
    res["weight"] = weight
    res["trainset"] = trainset
    res["testset"] = testset
    return res


def makeDataLoaders(trainset_path, testset_path):
    transform_train = Compose([ToTensor(), RandomHorizontalFlip()])
    transform_test = Compose([ToTensor()])
    dataset = torchvision.datasets.DatasetFolder
    trainset = dataset(
        root=os.path.join(trainset_path),
        loader=cv2.imread,
        extensions=("png",),
        transform=transform_train,
        target_transform=None,
        is_valid_file=None,
    )

    testset = dataset(
        root=testset_path,
        loader=cv2.imread,
        extensions=("png",),
        transform=transform_test,
        target_transform=None,
        is_valid_file=None,
    )
    return trainset, testset


def freeze_dataset_JSON(dataset_path):
    image_path_dict = {}
    if not f"trainset.json" in os.listdir():
        for single_class in os.listdir(dataset_path):
            tem_path = os.path.join(dataset_path, single_class)
            for idx, path in tqdm.tqdm(enumerate(os.listdir(tem_path))):
                image_path = os.path.join(tem_path, path)
                if (image_path[-4:]) == ".png":
                    image_np = cv2.imread(image_path)
                    data = cv2.imencode(".jpg", image_np)[1]
                    image_bytes = data.tobytes()
                    image_base64 = base64.b64encode(image_bytes).decode("utf8")
                    image_idx = image_path.split("/")
                    image_path_dict[image_base64] = os.path.join(
                        image_idx[-2], image_idx[-1]
                    )
        with open(f"trainset.json", "w") as fp:
            json.dump(image_path_dict, fp)
    else:
        with open(f"trainset.json", "r") as fp:
            image_path_dict = json.load(fp)
            print("Found prepared path dictionary, Loaded")
    return image_path_dict


def hp_json_write_file(path, content):
    data = json.dumps(content, indent=1)
    with open(path, "w", newline="\n") as f:
        f.write(data)


def json_write_file(path, content):
    with open(path, "w") as f:
        json.dump(content, f)


def json_read_file(path):
    with open(path, "r") as fp:
        parameter = json.load(fp)
    return parameter


# def get_borda_path(borda_list, ratio, index_to_path_dict, isSalient = False):
#     borda_list = torch.tensor(borda_list)
#     top_values, top_indices = torch.topk(borda_list, int(len(borda_list)*ratio), largest=isSalient)
#     dataloader_indexs = top_indices.tolist()
#     return [index_to_path_dict[str(index)] for index in dataloader_indexs]
def get_borda_path(
    borda_list,
    ratio,
    index_to_path_dict,
    isSalient=False,
    classes=10,
    poisoned_label="airplane",
    kmeans_k=1,
):
    # print("this is chosen label "+poisoned_label)
    total_count = len(borda_list) * ratio
    count_for_each_class = int(total_count / (classes - 1))
    k_means_dict = get_kmeans_result_cifar10(k=kmeans_k)
    count_for_each_cluster = analyze_cluster(k_means_dict, ratio, count_for_each_class)
    residual = total_count - count_for_each_class * (classes - 1)
    full_class = 0
    score_to_index_dict = {}
    path_to_ret = []
    score_to_ret = []
    for index in range(len(borda_list)):
        if borda_list[index] not in score_to_index_dict:
            score_to_index_dict[borda_list[index]] = [index]
        else:
            score_to_index_dict[borda_list[index]].append(index)
    sorted_borda = sorted(borda_list, reverse=isSalient)
    # print(sorted_borda)
    voted_separate_dict = {}

    for score in sorted_borda:
        index = score_to_index_dict[score][0]
        score_to_index_dict[score] = score_to_index_dict[score][1:]
        path = index_to_path_dict[str(index)]

        path: str
        if full_class == classes - 1:
            if residual > 0:
                residual -= 1
                path_to_ret.append(path)
                score_to_ret.append(score)
            else:
                break
        corres_class = path.split("/")[-2]
        """
            'airplane', 'automobile', 'bird', 0-2
            'cat', 'deer', 'dog', 3-5
            'frog', 'horse', 'ship', 'truck' 5-9
        """
        cluster_index = get_cluster_idx(k_means_dict, path, corres_class)
        # print(corres_class)
        if corres_class == poisoned_label:
            continue
        if (
            corres_class not in voted_separate_dict
            and count_for_each_cluster[corres_class][cluster_index] != 0
        ):
            voted_separate_dict[corres_class] = count_for_each_class - 1
            count_for_each_cluster[corres_class][cluster_index] -= 1
            path_to_ret.append(path)
            score_to_ret.append(score)
        elif (
            voted_separate_dict[corres_class] == 0
            or count_for_each_cluster[corres_class][cluster_index] == 0
        ):
            continue
        else:
            count_for_each_cluster[corres_class][cluster_index] -= 1
            voted_separate_dict[corres_class] = voted_separate_dict[corres_class] - 1
            path_to_ret.append(path)
            score_to_ret.append(score)
            if voted_separate_dict[corres_class] == 0:
                full_class += 1
        # print(score_to_ret)
    # print(path_to_ret[:10])
    return path_to_ret, score_to_ret


# def get_cluster_dict(clusters):
#     res_dict = {}
#     for idx, cluster in clusters:
#         res_dict[str(idx)] = cluster
#     return res_dict


def analyze_cluster(cluster_dict, ratio, count_for_each_class):
    ret_dict = {}
    for key, value in cluster_dict.items():
        quota_list = []
        for cluster in value:
            quota_list.append(math.ceil(len(cluster) * ratio))
        ret_dict[key] = quota_list
    # for key, value in ret_dict.items():
    #     class_total = count_for_each_class
    #     for num in value:
    #         class_total -= num
    #     while class_total != 0:
    #         if class_total > 0:
    #             class_total -= 1
    #             value[class_total % len(value)] += 1
    #         elif class_total < 0:
    #             class_total += 1
    #             value[class_total % len(value)] -= 1
    # print(ret_dict)
    # for key, value in ret_dict:
    #     print(key, value)
    return ret_dict


def get_cluster_idx(cluster_dict, path, corres_class):
    target = cluster_dict[corres_class]
    for idx, cluster in enumerate(target):
        if path in cluster:
            # print(path,idx)
            return idx
    return None


def get_dataset_size(path):
    return len(os.listdir(path))


def visualize_image(path_list, score_list, total):
    image_list = []
    for path in path_list:
        # Load the image using OpenCV
        opencv_image = cv2.imread(path)
        # Convert the OpenCV image to RGB
        opencv_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        # Define the transformation to convert the OpenCV image to a tensor
        transform = transforms.ToTensor()
        # Apply the transformation to the OpenCV image
        tensor_image = transform(opencv_image_rgb)
        reshaped_tensor = tensor_image.permute(1, 2, 0)
        image_list.append(reshaped_tensor)

    nrows = math.ceil(math.sqrt(total))
    ncols = math.ceil(total / nrows)
    plt.style.use("ggplot")
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(32, 32))

    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]

        if i < len(image_list):
            # Get the current image tensor
            image = image_list[i]
            # Convert the image tensor to a NumPy array
            image_np = image.numpy()
            # Plot the image in the subplot
            ax.imshow(image_np)
            # Adding the score as a caption to the image
            ax.set_title(f"Score: {score_list[i]}", fontsize=10)
        ax.axis("off")

    # Adjust the spacing between subplots
    plt.tight_layout()

    figure_path = "voted_image.svg"
    # Save the plot
    plt.savefig(figure_path, format="svg", transparent=True)
    print("Selection all clear,", f"visualized figure saved at: {figure_path}")


def map_possibility_to_rank(possibility_list):
    possibility_list: list
    sorted_list = sorted(possibility_list)
    mapping_dict = {}
    rank = 0
    for possibility in sorted_list:
        if possibility not in mapping_dict:
            mapping_dict[possibility] = rank
        rank += 1
    possibility_list_copy = [0 for i in range(len(possibility_list))]
    for index in range(len(possibility_list)):
        possibility_list_copy[index] = mapping_dict[possibility_list[index]]
    return possibility_list_copy


def borda_count(list_of_lists):
    initial = np.asarray(list_of_lists[0]) * 0
    for i in list_of_lists:
        to_add = map_possibility_to_rank(i)
        initial = initial + np.asarray(to_add)
    return initial / len(list_of_lists)


def checkCorrect_and_get_possibility(logit, label):
    predicted_correct_index = torch.argmax(logit, dim=-1)
    # print(logit.shape)
    # print(label.shape)
    true_correct_index = label
    possibility = F.softmax(logit, dim=-1)
    list_possibility = [
        possibility[index][true_correct_index[index]].item()
        for index in range(possibility.size(0))
    ]
    # print(list_possibility)
    # print(np_possibility)
    return (predicted_correct_index == true_correct_index), list_possibility


"""

@ret: dictionary of clusters of different class, in each cluster, the address is been stored
"""


def get_kmeans_result_cifar10(dataset="../input0/cifar10/cifar10/train", k=10):
    # resnet50 = models.resnet50(pretrained=False)
    resnet50 = core.models.ResNet(50)
    model_name = "resnet50"
    # vgg.load_state_dict(torch.load("/output/benign_models/vgg16bn_experiment005/ckpt_epoch_200.pth"))
    resnet50.load_state_dict(
        torch.load("/output/benign_models/resnet50_experiment002/ckpt_epoch_200.pth")
    )
    net = torch.nn.Sequential(*list(resnet50.children())[:-1])
    net.to("cuda:0")
    net.eval()
    transform_ = Compose([ToTensor()])
    res_dict = {}
    classes = os.listdir(dataset)
    classes = sorted(classes)
    # print(classes)
    for class_index in range(len(classes)):
        class_name = classes[class_index]
        res_dict[class_name] = []
        img_path = []
        all_img_in_single_class = []
        for path in sorted(os.listdir(os.path.join(dataset, class_name))):
            img_path.append(os.path.join(class_name, path))
        tem = None
        batch_size = 250
        for idx, path in tqdm.tqdm(
            enumerate(img_path),
            total=len(img_path),
            desc=f"            Extracting {model_name} embeddings and voting for class {img_path[0].split('/')[0]}",
        ):
            img = cv2.imread(os.path.join(dataset, path))
            img = transform_(img)
            img = img.unsqueeze(0)
            if idx % batch_size == 0:
                tem = img
            else:
                tem = torch.cat((tem, img), dim=0)
            if (idx + 1) % batch_size == 0:
                avg_pooling = torch.nn.AdaptiveAvgPool2d((1, 1))
                output = avg_pooling(net(tem.to("cuda:0")))
                tem = output.view(batch_size, -1).cpu().detach().numpy()
                all_img_in_single_class[idx + 1 - batch_size : idx + 1] = [
                    i for i in tem
                ]
                tem = None

        clt, labelIDs = k_means_cluster(k, all_img_in_single_class)
        # clt, labelIDs = dbscan_means_cluster(
        #     eps_value=1.5,
        #     min_samples=3,
        #     dimension_reduction=32,
        #     all_img_in_single_class=all_img_in_single_class,
        # )

        print("Silhouette score:", silhouette_score(all_img_in_single_class, clt.labels_))

        file_path = "kMeans_visulization"
        if class_name in os.listdir(file_path):
            shutil.rmtree(os.path.join(file_path, class_name))
        for labelID in labelIDs:
            idxs = np.where(clt.labels_ == labelID)[0]
            res_dict[class_name].append(np.asarray(img_path)[np.asarray(idxs)])
            idxs = np.random.choice(idxs, size=min(100, len(idxs)), replace=False)
            show_box = []
            for i in idxs:
                image = cv2.imread(os.path.join(dataset, img_path[i]))
                image = cv2.resize(image, (48, 48))
                show_box.append(image)
            montage = build_montages(show_box, (48, 48), (10, 10))[0]
            if file_path not in os.listdir():
                os.mkdir(file_path)
            class_file = os.path.join(file_path, class_name)
            if class_name not in os.listdir(file_path):
                os.mkdir(class_file)
            cv2.imwrite(
                os.path.join(class_file, (class_name + str(labelID))) + ".png", montage
            )
            cv2.waitKey(0)
    # for key in res_dict.keys():
    #     print(key)
    # json_write_file(f"clusters.json", res_dict)
    return res_dict


def k_means_cluster(k, all_img_in_single_class):
    pca = PCA(n_components=0.99)
    clt = KMeans(n_clusters=k, init="k-means++")
    all_img_in_single_class = pca.fit_transform(all_img_in_single_class)
    # print(len(all_img_in_single_class[0]))
    # for consine distance, uncomment following:
    all_img_in_single_class = np.asarray(all_img_in_single_class)
    all_img_in_single_class = normalize(all_img_in_single_class)
    distances = pairwise_distances(all_img_in_single_class, metric="cosine")
    clt.fit(distances)

    # for euclidean, use following:
    # clt.fit(all_img_in_single_class)
    labelIDs = np.unique(clt.labels_)
    
    return clt, labelIDs


def dbscan_means_cluster(
    eps_value, min_samples, dimension_reduction, all_img_in_single_class
):
    # tsne = TSNE(n_components=dimension_reduction, random_state=0)
    # reduced_data = tsne.fit_transform(np.asarray(all_img_in_single_class))
    pca = PCA(n_components=dimension_reduction)
    all_img_in_single_class = pca.fit_transform(all_img_in_single_class)
    clt = DBSCAN(eps=eps_value, min_samples=min_samples)
    # for consine distance, uncomment following:
    # all_img_in_single_class = np.asarray(all_img_in_single_class)
    # print(all_img_in_single_class, all_img_in_single_class.size)
    # all_img_in_single_class = normalize(all_img_in_single_class)
    # print(all_img_in_single_class, all_img_in_single_class.size)
    # distances = pairwise_distances(all_img_in_single_class, metric='cosine')
    # for euclidean, use following:
    clt.fit(all_img_in_single_class)
    labelIDs = np.unique(clt.labels_)
    return clt, labelIDs


# if __name__ == "__main__":
#     # input/input0/cifar10/cifar10/train
#     # split_dataset("../input0/cifar10/cifar10/train",5,"cifar10","TRAINSUB")
#     # print(1)
#     # The following is just for testing the log function
#     schedule = {
#         'device': 'GPU',
#         'CUDA_VISIBLE_DEVICES': '0',
#         'GPU_num': 1,

#         'benign_training': True,
#         'batch_size': 128,
#         'num_workers': 16,

#         'lr': 0.1,
#         'momentum': 0.9,
#         'weight_decay': 5e-4,
#         'gamma': 0.1,
#         'schedule': [40, 60],
#         'model_type': "resnet18",

#         'epochs': 70,

#         'log_iteration_interval': 100,
#         'test_epoch_interval': 10,
#         'save_epoch_interval': 20,

#         'save_dir': "experiment",
#         'experiment_name': 'train_benign_DatasetFolder-CIFAR10'
#     }


#     trainset, testset = makeDataLoaders(os.path.join("/output/TRAINSUB/SET0"),"../input0/cifar10/cifar10/test")

#     make_benign_model(trainset, schedule, testset)
# test1 = [4,2,41,1,1,1,1,1]
# test0 = map_possibility_to_rank(test1)
# print(test1)
# print(test0)

# def makeLoaderForClass(className):

#     transform_ = Compose([
#     Resize([224,224]),
#     ToTensor()
#     ])
#     trainset = torchvision.datasets.DatasetFolder(
#     root=os.path.join(dataset,className),
#     loader=cv2.imread,
#     extensions=('png',),
#     transform=transform_,
#     target_transform=None,
#     is_valid_file=None)

#     train_loader = DataLoader(
#             trainset,
#             batch_size=1000,
#             shuffle=False,
#             num_workers=24,
#             drop_last=False,
#             pin_memory=True,
#         )
#     return train_loader

# def get_borda_path(borda_list, ratio, index_to_path_dict, isSalient = False, classes = 10,
#                poisoned_label="airplane",kmeans_k=1):
# # print("this is chosen label "+poisoned_label)

# total_count = len(borda_list) * ratio
# count_for_each_class = int(total_count/(classes-1))
# residual = total_count-count_for_each_class*(classes-1)
# full_class = 0
# score_to_index_dict = {}
# to_ret = []
# for index in range(len(borda_list)):
#     if borda_list[index] not in score_to_index_dict:
#         score_to_index_dict[borda_list[index]] = [index]
#     else:
#         score_to_index_dict[borda_list[index]].append(index)
# sorted_borda = sorted(borda_list,reverse=isSalient)
# # print(sorted_borda)
# voted_separate_dict = {}
# for score in sorted_borda:
#     index = score_to_index_dict[score][0]
#     score_to_index_dict[score] = score_to_index_dict[score][1:]
#     path = index_to_path_dict[str(index)]
#     path:str
#     if full_class == classes-1:
#         if residual >0:
#             residual -= 1
#             to_ret.append(path)
#         else:
#             break
#     corres_class = path.split("/")[-2]
#     # print(corres_class)
#     if corres_class == poisoned_label:
#         continue
#     if corres_class not in voted_separate_dict:
#         voted_separate_dict[corres_class] = count_for_each_class-1
#         to_ret.append(path)
#     elif voted_separate_dict[corres_class] == 0:
#         continue
#     else:
#         voted_separate_dict[corres_class] = voted_separate_dict[corres_class]-1
#         to_ret.append(path)
#         if voted_separate_dict[corres_class] == 0:
#             full_class += 1
#     # print(to_ret)
# return to_ret




#def set_seed(seed):
#    torch.manual_seed(seed)
#    np.random.seed(seed)
#    os.environ['PYTHONHASHSEED']=str(seed)
#    if torch.cuda.is_available():
#        torch.cuda.manual_seed_all(seed)
#        torch.backends.cudnn.deterministic=True
#        torch.backends.cudm.benchmark=False

#param_grid={
#    'eps_valu': list(rannge(0,150)),
#    'min_samples':list(rannge(0,50)),
#    'dimension_reduction':[],
#    }
#MAX_EVALS=100
#best_score=0
#best_hyperparams={}
#for i in range(MAX_EVALS):
#    random.seed(50)
#    hyperparameters={k:random.sample(v,1)[0] for k, v in param_grid.items()}
#    eps_valu=hyperparameters['eps_valu']
#    min_samples=hyperparameters['min_samples']
#    dimension_reduction=hyperparameters['dimension_reduction']
    
#    model=myModel(eps_valu,min_samples,dimension_reduction)
#    train_model(model)
#    score= evaluate(model)
#    if score>best_score:
#        best_hyperparams=hyperparameters
#        best_score=score
        
#        torch.save(model.state_dict(),"best_model.pt")
    

