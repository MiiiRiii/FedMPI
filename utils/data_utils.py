import torch
import torchvision
import random
import numpy as np

from utils.utils import printLog
from torch.utils.data import Dataset

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms.
       모든 데이터를 한 번에 부르지 않고 하나씩만 불러써 쓰는 방식
    """

    def __init__(self, tensors, transform=None):  # 데이터셋 전처리
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):  # 데이터셋에서 특정 1개의 샘플을 가져옴
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y

    def __len__(self):  # 총 샘플의 수
        return self.tensors[0].size(0)


def applyCustomDataset(dataset_name,data, label):
    
    if dataset_name in ["CIFAR10"] :
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
    elif dataset_name in ["MNIST"] or dataset_name in ["FashionMNIST"]:
            transform = torchvision.transforms.ToTensor()
    """
    elif dataset_name in ["FashionMNIST"]:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), torchvision.transforms.Resize((16,16))
        ])
    """
    return CustomTensorDataset((data, label.long()), transform=transform)

def get_local_datasets_labels_probabilities(datasets):
    len=0
    labels=[]
    labels_info={0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
    for data in datasets:
        labels_info[data[1].item()]+=1
        labels.append(data[1].item())
        len+=1
    label_set = set(labels)
    probabilities = [labels_info[element]/len for element in label_set]
    return label_set, probabilities

def get_uniform_mini_batch(dataset_name, local_datasets, uniform_random_labels, b=32):
    shuffled_indices = torch.randperm(len(local_datasets))
    shuffled_local_datasets = [local_datasets[idx] for idx in shuffled_indices]
    mini_batch_data = []
    mini_batch_label = []
    check_selected = [False for i in range(b)]
    cnt=0
    for data in shuffled_local_datasets:
       for l in range(b):
           if data[1].item() == uniform_random_labels[l] and check_selected[l]==False:
               check_selected[l]=True
               cnt+=1
               mini_batch_data.append(data[0])
               mini_batch_label.append(data[1])
               break
    uniform_mini_batch = applyCustomDataset(dataset_name, torch.cat(mini_batch_data,0), torch.tensor(mini_batch_label))
    return uniform_mini_batch

def create_uniform_labels(unique_labels, probabilities, b):
    uniform_random_labels = random.choices(unique_labels, probabilities, k=b)
    return uniform_random_labels

def create_dataset(num_clients, dataset_name, iid, split):
    num_shards=200
    data_path="./data/"
    iid = (iid=="True")
    if dataset_name in ["CIFAR10"] :
            transform = torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ]
            )
    elif dataset_name in ["MNIST"] or dataset_name in ["FashionMNIST"]:
            transform = torchvision.transforms.ToTensor()
    
    """
    elif dataset_name in ["FashionMNIST"]:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((16,16)), torchvision.transforms.ToTensor()
        ])
    """
    test_dataset = torchvision.datasets.__dict__[dataset_name](
        root=data_path,
        train=False,
        download=True,
        transform=transform
    )

    # raw training dataset을 지정된 data_path에 다운로드
    training_dataset = torchvision.datasets.__dict__[dataset_name](  # torchvision.dataset__dict__
        root=data_path,
        train=True,
        download=True,
        transform=transform
    )
    
    if training_dataset.data.ndim == 3 : # 3차원이라면 4차원으로 바꿈
        training_dataset.data.unsqueeze_(3) # convert to N x H x W -> N x H x W x 1 
    if "ndarray" not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset.data)
    if "list" not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
        
    # config.yaml에 있는 flag가 iid면 dataset을 iid로 split
    if iid:
        # shuffle data
        shuffled_indices = torch.randperm(
            len(training_dataset))  # data의 index를 셔플함
        training_inputs = training_dataset.data[shuffled_indices]
        training_labels = torch.Tensor(training_dataset.targets)[
            shuffled_indices]

        # client 개수만큼 data를 나눔
        split_size = len(training_dataset) // num_clients
        split_datasets = list(  # training data와 label을 묶어서 list로 만듦
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels.long()), split_size)
            )
        )
        # finalize bunches of local datasets
        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]
    # non-iid로 dataset split
    else:
        local_datasets = []

        # training_data의 label을 오름차순으로 정렬
        # 원래는 [6, 9, 9, 4, ... ] 이었던 list를 tensor([0., 0., 0.,  ..., 9., 9., 9.])로 변환
        # training_dataset의 label을 오름차순으로 정렬하여 해당 요소가 있는 index list를 반환
        sorted_indices = torch.argsort(
            torch.Tensor(training_dataset.targets))
        training_inputs = training_dataset.data[sorted_indices]
        training_labels = torch.Tensor(training_dataset.targets)[
            sorted_indices]  # tensor([0., 0., 0.,  ..., 9., 9., 9.])

        len_training_inputs = len(training_inputs)

        if split == "uniform":  # client들이 가져가는 데이터 개수가 모두 동일
            local_datasets = split_dataset_uniform(
                num_shards, training_inputs, training_labels, num_clients, transform, len_training_inputs)
        elif split == "gaussian":  # client들이 가져가는 데이터 개수가 gaussian 분포를 따름
            std=0
            if dataset_name=="CIFAR10":
                std=300
            elif dataset_name=="MNIST":
                std=400
            elif dataset_name=="FashionMNIST":
                std=400
            local_datasets = split_dataset_gaussian(
                training_inputs, training_labels, num_clients, transform, len_training_inputs, std)

    return local_datasets, test_dataset

def split_dataset_uniform(num_shards, training_inputs, training_labels, num_clients, transform, len_training_inputs):
    """
        1. 6만개의 training dataset을 label 기준 오름차순으로 정렬한다.
        2. 6만개의 training dataset을 크기가 300인 200개의 shard로 나눈다.
        3. 한 client에게 2개의 shard를 준다. (client 1개는 총 600개의 data를 가짐)
        client idx 순서대로 0,20,30,...,180,1,21,...,181,2,22,42..., ... 번째의 shard를 2개씩 가져감    
    """

    num_categories = np.unique(training_labels).shape[0]

    len_data_each_client = len_training_inputs // num_shards

    # shard_size 만큼 data를 split함
    shard_inputs = list(torch.split(
        torch.Tensor(training_inputs), len_data_each_client))
    # shard_size 만큼 label을 split함
    shard_labels = list(torch.split(
        torch.Tensor(training_labels), len_data_each_client))

    shard_inputs_sorted, shard_labels_sorted = [], []
    for i in range(num_shards // num_categories):
        # rang(a,b,c) : a부터 b-1까지 c만큼의 간격으로 값을 반환
        for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
            # 0, 20, 40, ..., 180, 1,21,41...,181, ... , ...
            shard_inputs_sorted.append(shard_inputs[i+j])
            shard_labels_sorted.append(shard_labels[i+j])

    # finalize local datasets by assigning shards to each client
    shards_per_clients = num_shards // num_clients
    local_datasets = [
        CustomTensorDataset(
            (
                torch.cat(shard_inputs_sorted[i:i+shards_per_clients]),
                torch.cat(
                    shard_labels_sorted[i:i+shards_per_clients]).long()
            ),
            transform=transform
        )
        for i in range(0, len(shard_inputs_sorted), shards_per_clients)
    ]
    return local_datasets


def split_dataset_gaussian(training_inputs, training_labels, num_clients, transform, len_training_inputs, std):
    """
    클라이언트들이 서로 다른 개수의 데이터를 나눠가진다. 클라이언트들이 나눠 가지는 데이터의 개수는 가우시안 분포를 따른다.
    """
    mean = len_training_inputs // num_clients
    len_training_dataset = len(training_labels)
    shard_size = get_dataset_index_by_gaussian(
        len_training_dataset, num_clients, mean, std)

    # shard_size 만큼 data를 split함
    shard_inputs = list(torch.split(
        torch.Tensor(training_inputs), shard_size))
    # shard_size 만큼 label을 split함
    shard_labels = list(torch.split(
        torch.Tensor(training_labels), shard_size))

    local_datasets = [
        CustomTensorDataset(
            (
                shard_inputs[i], shard_labels[i].long()
            ),
            transform=transform
        )
        for i in range(num_clients)
    ]

    return local_datasets


def get_dataset_index_by_gaussian(len_training_dataset, num_clients, mean, std):
    """
    평균은 mean이고, 표준편차는 std인 가우시안 분포를 따르는 num_clients개의 정수 list를 반환한다.
    이때 num_clients개의 정수들의 합은 len_training_dataset과 동일하다.
    """
    result = []
    while True:
        result.clear()
        current_allocated_amount = 0

        success_flag = True
        for b in range(num_clients-1):
            sampled_value = positive_int_gauss_random(mean, std)
            current_allocated_amount += sampled_value
            if current_allocated_amount >= len_training_dataset:
                success_flag = False
                break
            result.append(sampled_value)

        if success_flag:
            break

    result.append(len_training_dataset-current_allocated_amount)

    printLog(f"각 client들이 가져갈 data 개수 = {result}")
    return result

def positive_int_gauss_random(mean, std):
    """
    평균은 mean이고, 표준편차는 std인 가우시안 분포를 따르는 랜덤 양수 하나를 생성한다.
    """
    sampled_value = -1
    while sampled_value <= 9:
        sampled_value = round(random.gauss(mean, std))

    return sampled_value
