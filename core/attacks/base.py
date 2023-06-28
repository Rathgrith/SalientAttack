import os
import os.path as osp
import time
import seaborn as sns
import matplotlib.pyplot as plt
from copy import deepcopy
import random
import cv2
import numpy as np
from torchvision.transforms import Compose, ToTensor, PILToTensor, RandomHorizontalFlip
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import DatasetFolder, MNIST, CIFAR10
from ..utils import Log
import base64
import json
import re
from ..utils.utility import (
    checkCorrect_and_get_possibility,
    json_read_file,
    json_write_file,
    hp_json_write_file,
)
import torch.nn.functional as F
import tqdm
import sys
import matplotlib.ticker as mticker
import torch.distributed as dist
import torch.multiprocessing as mp
import argparse

support_list = (DatasetFolder, MNIST, CIFAR10)

testset_top1_correct = []
poisoned_set_top1_correct = []


def check(dataset):
    return isinstance(dataset, support_list)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    #     print(correct)
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class Base(object):
    """Base class for backdoor training and testing.

    Args:
        train_dataset (types in support_list): Benign training dataset.
        test_dataset (types in support_list): Benign testing dataset.
        model (torch.nn.Module): Network.
        loss (torch.nn.Module): Loss.
        schedule (dict): Training or testing global schedule. Default: None.
        seed (int): Global seed for random numbers. Default: 0.
        deterministic (bool): Sets whether PyTorch operations must use "deterministic" algorithms.
            That is, algorithms which, given the same input, and when run on the same software and hardware,
            always produce the same output. When enabled, operations will use deterministic algorithms when available,
            and if only nondeterministic algorithms are available they will throw a RuntimeError when called. Default: False.
    """

    def __init__(
        self,
        train_dataset,
        test_dataset,
        model,
        loss,
        schedule=None,
        seed=0,
        deterministic=False,
    ):
        assert isinstance(
            train_dataset, support_list
        ), "train_dataset is an unsupported dataset type, train_dataset should be a subclass of our support list."
        self.train_dataset = train_dataset

        assert isinstance(
            test_dataset, support_list
        ), "test_dataset is an unsupported dataset type, test_dataset should be a subclass of our support list."
        self.test_dataset = test_dataset
        self.model = model
        self.loss = loss
        self.global_schedule = deepcopy(schedule)
        self.current_schedule = None
        self._set_seed(seed, deterministic)
        self.experiment_id = 0
        self.muti_gpus = False

    def _set_seed(self, seed, deterministic):
        # Use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA).
        torch.manual_seed(seed)

        # Set python seed
        random.seed(seed)

        # Set numpy seed (However, some applications and libraries may use NumPy Random Generator objects,
        # not the global RNG (https://numpy.org/doc/stable/reference/random/generator.html), and those will
        # need to be seeded consistently as well.)
        np.random.seed(seed)

        os.environ["PYTHONHASHSEED"] = str(seed)

        if deterministic:
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)
            torch.backends.cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            # Hint: In some versions of CUDA, RNNs and LSTM networks may have non-deterministic behavior.
            # If you want to set them deterministic, see torch.nn.RNN() and torch.nn.LSTM() for details and workarounds.

    def _seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_model(self):
        return self.model

    def get_poisoned_dataset(self):
        return self.poisoned_train_dataset, self.poisoned_test_dataset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch in self.current_schedule["schedule"]:
            self.current_schedule["lr"] *= self.current_schedule["gamma"]
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.current_schedule["lr"]

    def train(self, schedule=None):
        if schedule is None and self.global_schedule is None:
            raise AttributeError(
                "Training schedule is None, please check your schedule setting."
            )
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)

        model_type = self.current_schedule["model_type"]
        existing_ids = []
        os.makedirs(self.current_schedule["save_dir"], exist_ok=True)
        for dir in os.listdir(self.current_schedule["save_dir"]):
            match = re.match(rf"{model_type}_experiment(\d+)", dir)
            if match:
                existing_ids.append(int(match.group(1)))
        if existing_ids:
            self.experiment_id = max(existing_ids) + 1
        else:
            self.experiment_id = 1
        formatted_experiment_id = f"{model_type}_experiment{self.experiment_id:03}"

        if "pretrain" in self.current_schedule:
            self.model.load_state_dict(
                torch.load(self.current_schedule["pretrain"]), strict=False
            )

        # Use GPU
        if (
            "device" in self.current_schedule
            and self.current_schedule["device"] == "GPU"
        ):
            if "CUDA_VISIBLE_DEVICES" in self.current_schedule:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.current_schedule[
                    "CUDA_VISIBLE_DEVICES"
                ]

            assert torch.cuda.device_count() > 0, "This machine has no cuda devices!"
            assert (
                self.current_schedule["GPU_num"] > 0
            ), "GPU_num should be a positive integer"
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train."
            # )

            if self.current_schedule["GPU_num"] == 1:
                device = torch.device("cuda:0")
            else:
                # gpus = list(range(self.current_schedule['GPU_num']))
                # self.model = nn.DataParallel(self.model.cuda(), device_ids=gpus, output_device=gpus[0])
                # # TODO: DDP training
                # pass
                self.muti_gpus = True
        # Use CPU
        else:
            device = torch.device("cpu")

        if self.muti_gpus == False:
            experiment_type = "benign"
            if self.current_schedule["benign_training"] is True:
                train_loader = DataLoader(
                    self.train_dataset,
                    batch_size=self.current_schedule["batch_size"],
                    shuffle=True,
                    num_workers=self.current_schedule["num_workers"],
                    drop_last=False,
                    pin_memory=True,
                    worker_init_fn=self._seed_worker,
                )
                experiment_type = "benign"
            elif self.current_schedule["benign_training"] is False:
                train_loader = DataLoader(
                    self.poisoned_train_dataset,
                    batch_size=self.current_schedule["batch_size"],
                    shuffle=True,
                    num_workers=self.current_schedule["num_workers"],
                    drop_last=False,
                    pin_memory=True,
                    worker_init_fn=self._seed_worker,
                )
                experiment_type = "poison"
            else:
                raise AttributeError(
                    "self.current_schedule['benign_training'] should be True or False."
                )

            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.current_schedule["lr"],
                momentum=self.current_schedule["momentum"],
                weight_decay=self.current_schedule["weight_decay"],
            )

            # work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
            work_dir = osp.join(
                self.current_schedule["save_dir"], formatted_experiment_id
            )
            os.makedirs(work_dir, exist_ok=True)
            log = Log(osp.join(work_dir, "log.txt"))
            experiment_log = osp.join(
                self.current_schedule["save_dir"], "experiment_log.txt"
            )
            hp_json_write_file(work_dir + "/hyperparameter.json", schedule)
            with open(experiment_log, "a") as f:
                f.write(
                    f"Experiment ID: {self.experiment_id}, Experiment Name: {self.current_schedule['experiment_name']}, Model Type: {model_type}\n"
                )
            self.model = self.model.to(device)
            self.model.train()
            # log and output:
            # 1. output loss and time
            # 2. test and output statistics
            # 3. save checkpoint

            iteration = 0
            last_time = time.time()

            msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
            log(msg)

            for i in range(self.current_schedule["epochs"]):
                self.adjust_learning_rate(optimizer, i)
                for batch_id, batch in enumerate(train_loader):
                    batch_img = batch[0]
                    batch_label = batch[1]
                    batch_img = batch_img.to(device)
                    batch_label = batch_label.to(device)
                    optimizer.zero_grad()
                    predict_digits = self.model(batch_img)
                    loss = self.loss(predict_digits, batch_label)
                    loss.backward()
                    optimizer.step()
                    iteration += 1

                    if iteration % self.current_schedule["log_iteration_interval"] == 0:
                        msg = (
                            time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                            + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                        )
                        last_time = time.time()
                        log(msg)

                if (i + 1) % self.current_schedule["test_epoch_interval"] == 0:
                    # test result on benign test dataset
                    predict_digits, labels = self._test(
                        self.test_dataset,
                        device,
                        self.current_schedule["batch_size"],
                        self.current_schedule["num_workers"],
                    )
                    total_num = labels.size(0)
                    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                    top1_correct = int(round(prec1.item() / 100.0 * total_num))
                    top5_correct = int(round(prec5.item() / 100.0 * total_num))
                    top1_accuracy = top1_correct / total_num
                    testset_top1_correct.append(top1_accuracy)
                    msg = (
                        "==========Test result on benign test dataset==========\n"
                        + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                        + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_accuracy}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                    )
                    log(msg)

                    # test result on poisoned test dataset
                    # if self.current_schedule['benign_training'] is False:
                    predict_digits, labels = self._test(
                        self.poisoned_test_dataset,
                        device,
                        self.current_schedule["batch_size"],
                        self.current_schedule["num_workers"],
                    )
                    total_num = labels.size(0)
                    prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                    top1_correct = int(round(prec1.item() / 100.0 * total_num))
                    top5_correct = int(round(prec5.item() / 100.0 * total_num))
                    top1_accuracy = top1_correct / total_num
                    poisoned_set_top1_correct.append(top1_accuracy)
                    msg = (
                        "==========Test result on poisoned test dataset==========\n"
                        + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                        + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_accuracy}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                    )
                    log(msg)

                    self.model = self.model.to(device)
                    self.model.train()
                    plt.figure(figsize=(10, 6))
                    plt.style.use("ggplot")
                    # print(testset_top1_correct, poisoned_set_top1_correct)
                    plt.plot(poisoned_set_top1_correct, "o-", label="Poisoned Test Set")
                    plt.plot(testset_top1_correct, "o-", label="Benign Test Set")
                    plt.xlabel("Epochs")
                    plt.ylabel("Top-1 Accuracy")
                    # xtick_labels = ['A', 'B', 'C', 'D', 'E']
                    span = self.current_schedule["test_epoch_interval"]
                    # interval = 5  # Set the interval according to your preference
                    plt.xticks(
                        range(0, len(testset_top1_correct), 1),
                        range(span, len(testset_top1_correct) * (span) + span, span),
                    )
                    myLocator = mticker.MultipleLocator(4)
                    plt.gca().xaxis.set_major_locator(myLocator)

                    plt.title("Training_log")
                    plt.legend()
                    figure_path = osp.join(work_dir, "training_log.svg")
                    plt.savefig(figure_path, format="svg")
                    print(f"Figure saved at: {figure_path}")

                if (i + 1) % self.current_schedule["save_epoch_interval"] == 0:
                    self.model.eval()
                    self.model = self.model.cpu()
                    ckpt_model_filename = "ckpt_epoch_" + str(i + 1) + ".pth"
                    ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                    torch.save(self.model.state_dict(), ckpt_model_path)
                    self.model = self.model.to(device)
                    self.model.train()
        elif self.muti_gpus == True:
            parser = argparse.ArgumentParser()
            args = parser.parse_args()
            args.nprocs = torch.cuda.device_count()
            mp.spawn(muti_gpu_worker, args.nprocs, args=(args.nprocs, args))

            def muti_gpu_worker(local_rank, nprocs):
                dist.init_process_group(
                    backend="nccl",
                    init_method="tcp://127.0.0.1:23456",
                    world_size=nprocs,
                    local_rank=local_rank,
                )
                torch.cuda.set_device(local_rank)

                train_sampler = DistributedSampler(self.train_dataset)
                if self.current_schedule["benign_training"] is True:
                    train_loader = DataLoader(
                        self.train_dataset,
                        batch_size=self.current_schedule["batch_size"],
                        shuffle=True,
                        num_workers=self.current_schedule["num_workers"],
                        drop_last=False,
                        pin_memory=True,
                        worker_init_fn=self._seed_worker,
                        sampler=train_sampler,
                    )
                elif self.current_schedule["benign_training"] is False:
                    train_loader = DataLoader(
                        self.poisoned_train_dataset,
                        batch_size=self.current_schedule["batch_size"],
                        shuffle=True,
                        num_workers=self.current_schedule["num_workers"],
                        drop_last=False,
                        pin_memory=True,
                        worker_init_fn=self._seed_worker,
                        sampler=train_sampler,
                    )
                    experiment_type = "poison"
                else:
                    raise AttributeError(
                        "self.current_schedule['benign_training'] should be True or False."
                    )
                self.model = nn.parallel.DistributedDataParallel(
                    self.model, device_ids=[args.local_rank]
                )
                optimizer = torch.optim.SGD(
                    self.model.parameters(),
                    lr=self.current_schedule["lr"],
                    momentum=self.current_schedule["momentum"],
                    weight_decay=self.current_schedule["weight_decay"],
                )

                # work_dir = osp.join(self.current_schedule['save_dir'], self.current_schedule['experiment_name'] + '_' + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()))
                work_dir = osp.join(
                    self.current_schedule["save_dir"], formatted_experiment_id
                )
                os.makedirs(work_dir, exist_ok=True)
                log = Log(osp.join(work_dir, "log.txt"))
                experiment_log = osp.join(
                    self.current_schedule["save_dir"], "experiment_log.txt"
                )
                hp_json_write_file(work_dir + "/hyperparameter.json", schedule)
                with open(experiment_log, "a") as f:
                    f.write(
                        f"Experiment ID: {self.experiment_id}, Experiment Name: {self.current_schedule['experiment_name']}, Model Type: {model_type}\n"
                    )
                self.model = self.model.to(device)
                self.model.train()
                # log and output:
                # 1. output loss and time
                # 2. test and output statistics
                # 3. save checkpoint

                iteration = 0
                last_time = time.time()

                msg = f"Total train samples: {len(self.train_dataset)}\nTotal test samples: {len(self.test_dataset)}\nBatch size: {self.current_schedule['batch_size']}\niteration every epoch: {len(self.train_dataset) // self.current_schedule['batch_size']}\nInitial learning rate: {self.current_schedule['lr']}\n"
                log(msg)

                for i in range(self.current_schedule["epochs"]):
                    self.adjust_learning_rate(optimizer, i)
                    for batch_id, batch in enumerate(train_loader):
                        batch_img = batch[0]
                        batch_label = batch[1]
                        batch_img = batch_img.to(device)
                        batch_label = batch_label.to(device)
                        optimizer.zero_grad()
                        predict_digits = self.model(batch_img)
                        loss = self.loss(predict_digits, batch_label)
                        loss.backward()
                        optimizer.step()
                        iteration += 1

                        if (
                            iteration % self.current_schedule["log_iteration_interval"]
                            == 0
                        ):
                            msg = (
                                time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                                + f"Epoch:{i+1}/{self.current_schedule['epochs']}, iteration:{batch_id + 1}/{len(self.poisoned_train_dataset)//self.current_schedule['batch_size']}, lr: {self.current_schedule['lr']}, loss: {float(loss)}, time: {time.time()-last_time}\n"
                            )
                            last_time = time.time()
                            log(msg)

                    if (i + 1) % self.current_schedule["test_epoch_interval"] == 0:
                        # test result on benign test dataset
                        predict_digits, labels = self._test(
                            self.test_dataset,
                            device,
                            self.current_schedule["batch_size"],
                            self.current_schedule["num_workers"],
                        )
                        total_num = labels.size(0)
                        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                        top1_correct = int(round(prec1.item() / 100.0 * total_num))
                        top5_correct = int(round(prec5.item() / 100.0 * total_num))
                        top1_accuracy = top1_correct / total_num
                        testset_top1_correct.append(top1_accuracy)
                        msg = (
                            "==========Test result on benign test dataset==========\n"
                            + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                            + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_accuracy}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                        )
                        log(msg)

                        # test result on poisoned test dataset
                        # if self.current_schedule['benign_training'] is False:
                        predict_digits, labels = self._test(
                            self.poisoned_test_dataset,
                            device,
                            self.current_schedule["batch_size"],
                            self.current_schedule["num_workers"],
                        )
                        total_num = labels.size(0)
                        prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
                        top1_correct = int(round(prec1.item() / 100.0 * total_num))
                        top5_correct = int(round(prec5.item() / 100.0 * total_num))
                        top1_accuracy = top1_correct / total_num
                        poisoned_set_top1_correct.append(top1_accuracy)
                        msg = (
                            "==========Test result on poisoned test dataset==========\n"
                            + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                            + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_accuracy}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
                        )
                        log(msg)

                        self.model = self.model.to(device)
                        self.model.train()
                        plt.figure(figsize=(10, 6))
                        plt.style.use("ggplot")
                        # print(testset_top1_correct, poisoned_set_top1_correct)
                        plt.plot(
                            poisoned_set_top1_correct, "o-", label="Poisoned Test Set"
                        )
                        plt.plot(testset_top1_correct, "o-", label="Benign Test Set")
                        plt.xlabel("Epochs")
                        plt.ylabel("Top-1 Accuracy")
                        # xtick_labels = ['A', 'B', 'C', 'D', 'E']
                        span = self.current_schedule["test_epoch_interval"]
                        # interval = 5  # Set the interval according to your preference
                        plt.xticks(
                            range(0, len(testset_top1_correct), 1),
                            range(
                                span, len(testset_top1_correct) * (span) + span, span
                            ),
                        )
                        myLocator = mticker.MultipleLocator(4)
                        plt.gca().xaxis.set_major_locator(myLocator)

                        plt.title("Training_log")
                        plt.legend()
                        figure_path = osp.join(work_dir, "training_log.svg")
                        plt.savefig(figure_path, format="svg")
                        print(f"Figure saved at: {figure_path}")

                    if (i + 1) % self.current_schedule["save_epoch_interval"] == 0:
                        self.model.eval()
                        self.model = self.model.cpu()
                        ckpt_model_filename = "ckpt_epoch_" + str(i + 1) + ".pth"
                        ckpt_model_path = os.path.join(work_dir, ckpt_model_filename)
                        torch.save(self.model.state_dict(), ckpt_model_path)
                        self.model = self.model.to(device)
                        self.model.train()

    def _test(
        self,
        dataset,
        device,
        batch_size=16,
        num_workers=8,
        model=None,
        image_title_dict=None,
    ):
        if model is None:
            model = self.model
        else:
            model = model

        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            model = model.to(device)
            model.eval()

            predict_digits = []
            labels = []
            for batch in test_loader:
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                labels.append(batch_label)
            predict_digits = torch.cat(predict_digits, dim=0)
            labels = torch.cat(labels, dim=0)
            return predict_digits, labels

    def test_setup(
        self, schedule, model, test_dataset=None, poisoned_test_dataset=None
    ):
        if schedule is None and self.global_schedule is None:
            raise AttributeError(
                "Test schedule is None, please check your schedule setting."
            )
        elif schedule is not None and self.global_schedule is None:
            self.current_schedule = deepcopy(schedule)
        elif schedule is None and self.global_schedule is not None:
            self.current_schedule = deepcopy(self.global_schedule)
        elif schedule is not None and self.global_schedule is not None:
            self.current_schedule = deepcopy(schedule)
        #         if "image_title_dict" in self.current_schedule:
        #             image_title_dict = self.current_schedule["image_title_dict"]
        if model is None:
            model = self.model

        if "test_model" in self.current_schedule:
            model.load_state_dict(
                torch.load(self.current_schedule["test_model"]), strict=False
            )

        if test_dataset is None and poisoned_test_dataset is None:
            test_dataset = self.test_dataset
            poisoned_test_dataset = self.poisoned_test_dataset

        # Use GPU
        if (
            "device" in self.current_schedule
            and self.current_schedule["device"] == "GPU"
        ):
            if "CUDA_VISIBLE_DEVICES" in self.current_schedule:
                os.environ["CUDA_VISIBLE_DEVICES"] = self.current_schedule[
                    "CUDA_VISIBLE_DEVICES"
                ]

            assert torch.cuda.device_count() > 0, "This machine has no cuda devices!"
            assert (
                self.current_schedule["GPU_num"] > 0
            ), "GPU_num should be a positive integer"
            # print(
            #     f"This machine has {torch.cuda.device_count()} cuda devices, and use {self.current_schedule['GPU_num']} of them to train."
            # )

            if self.current_schedule["GPU_num"] == 1:
                device = torch.device("cuda:0")
            else:
                gpus = list(range(self.current_schedule["GPU_num"]))
                model = nn.DataParallel(
                    model.cuda(), device_ids=gpus, output_device=gpus[0]
                )
                # TODO: DDP training
                pass
        # Use CPU
        else:
            device = torch.device("cpu")
        return device

    def test(
        self,
        schedule=None,
        model=None,
        test_dataset=None,
        poisoned_test_dataset=None,
        image_title_dict=None,
    ):
        device = self.test_setup(schedule, model, test_dataset, poisoned_test_dataset)
        work_dir = osp.join(
            self.current_schedule["save_dir"],
            self.current_schedule["experiment_name"]
            + "_"
            + time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
        )
        os.makedirs(work_dir, exist_ok=True)
        log = Log(osp.join(work_dir, "log.txt"))

        if test_dataset is not None:
            last_time = time.time()
            # test result on benign test dataset
            predict_digits, labels = self._test(
                test_dataset,
                device,
                self.current_schedule["batch_size"],
                self.current_schedule["num_workers"],
                model,
                image_title_dict=image_title_dict,
            )
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top1_accuracy = top1_correct / total_num
            # testset_top1_correct.append(top1_accuracy)
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            msg = (
                "==========Test result on benign test dataset==========\n"
                + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_accuracy}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            )
            log(msg)

        if poisoned_test_dataset is not None:
            last_time = time.time()
            # test result on poisoned test dataset
            predict_digits, labels = self._test(
                poisoned_test_dataset,
                device,
                self.current_schedule["batch_size"],
                self.current_schedule["num_workers"],
                model,
            )
            total_num = labels.size(0)
            prec1, prec5 = accuracy(predict_digits, labels, topk=(1, 5))
            top1_correct = int(round(prec1.item() / 100.0 * total_num))
            top5_correct = int(round(prec5.item() / 100.0 * total_num))
            top1_accuracy = top1_correct / total_num
            # poisoned_set_top1_correct.append(top1_accuracy)
            msg = (
                "==========Test result on poisoned test dataset==========\n"
                + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}, Top-5 correct / Total: {top5_correct}/{total_num}, Top-5 accuracy: {top5_correct/total_num}, time: {time.time()-last_time}\n"
            )
            log(msg)

        """
        input: schedule: dict, index_mapping: bool
        output: Tuple[List[], dict(key: path, value: idx)]
        """

    def infer(
        self,
        schedule=None,
        model=None,
        test_datasets=[],
        poisoned_test_dataset=None,
        image_title_dict=None,
        train_index=0,
    ):
        device = self.test_setup(schedule, model, test_datasets, poisoned_test_dataset)
        # predict_list = []
        # image_pred_dict = {}
        index_to_path_dict = {}
        all_paths = []
        all_possibility_on_true_label = []

        # check if it is necessary to calculate the p_i_dict
        to_calculate_p_i_dict = False
        if not f"idx_to_path.json" in os.listdir():
            to_calculate_p_i_dict = True
        if test_datasets is not None:
            for dataset_index in range(len(test_datasets)):
                dataset = test_datasets[dataset_index]
                last_time = time.time()
                # test result on benign test dataset
                transform_test = Compose([ToTensor()])
                dataset_folder = DatasetFolder
                dataset = dataset_folder(
                    root=dataset,
                    # here the path will become the dataset folder object
                    loader=cv2.imread,
                    extensions=("png",),
                    transform=transform_test,
                    target_transform=None,
                    is_valid_file=None,
                )

                predict_digits, labels, labelled_possibility, paths = self._infer(
                    dataset,
                    device,
                    schedule["batch_size"],
                    self.current_schedule["num_workers"],
                    model,
                    image_title_dict=image_title_dict,
                    to_calculate_p_i_dict=to_calculate_p_i_dict,
                )
                all_paths.extend(paths)
                if dataset_index == train_index:
                    labelled_possibility = np.asarray(labelled_possibility) * 0
                    # predict_digits = torch.tensor([0 for i in range(len(dataset))])
                    # labels = torch.tensor([-1 for i in range(len(dataset))])
                all_possibility_on_true_label.extend(labelled_possibility)
                total_num = labels.size(0)
                prec1 = accuracy(predict_digits, labels)[0]
                top1_correct = int(round(prec1.item() / 100.0 * total_num))
                # top5_correct = int(round(prec5.item() / 100.0 * total_num))
                msg = (
                    "==========Test result on benign test dataset==========\n"
                    + time.strftime("[%Y-%m-%d_%H:%M:%S] ", time.localtime())
                    + f"Top-1 correct / Total: {top1_correct}/{total_num}, Top-1 accuracy: {top1_correct/total_num}\n"
                )
                # log(msg)
                # print(msg)
            if to_calculate_p_i_dict:
                for i in range(len(all_paths)):
                    index_to_path_dict[i] = all_paths[i]
                json_write_file("idx_to_path.json", index_to_path_dict)
            else:
                index_to_path_dict = json_read_file("idx_to_path.json")
        else:
            raise AttributeError("infer datasets can not be none")
        # print(len(all_possibility_on_true_label))
        return all_possibility_on_true_label

    def _infer(
        self,
        dataset,
        device,
        batch_size=1,
        num_workers=8,
        model=None,
        image_title_dict=None,
        to_calculate_p_i_dict=False,
    ):
        if model is None:
            model = self.model
        else:
            model = model
        with torch.no_grad():
            test_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                drop_last=False,
                pin_memory=True,
                worker_init_fn=self._seed_worker,
            )

            model = model.to(device)
            model.eval()

            labeled_possibility = []
            predict_digits = []
            labels = []
            path = []
            for idx, batch in enumerate(test_loader):
                batch_img, batch_label = batch
                batch_img = batch_img.to(device)
                if to_calculate_p_i_dict:
                    image_base64 = self.image_tensor_to_base64(batch_img)
                    path.append(image_title_dict[image_base64])
                batch_img = model(batch_img)
                batch_img = batch_img.cpu()
                predict_digits.append(batch_img)
                isCorrect, correct_possibility = checkCorrect_and_get_possibility(
                    batch_img, batch_label
                )
                # print(correct_possibility.shape)
                labeled_possibility.extend(correct_possibility)
                labels.append(batch_label)
            predict_digits = torch.cat(predict_digits, dim=0)
            # print(labels.shape)
            labels = torch.cat(labels, dim=0)
        return predict_digits, labels, labeled_possibility, path

    def image_tensor_to_base64(self, batch_img):
        batch_image_reshaped = batch_img.permute(0, 2, 3, 1)
        image_np = batch_image_reshaped[0].cpu().numpy()
        image_np = np.uint8(image_np * 255)
        data = cv2.imencode(".jpg", image_np)[1]
        image_bytes = data.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf8")
        return image_base64
