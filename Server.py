import torch.distributed as dist
import torch

from utils.utils import *
from utils.model_utils import TensorBuffer
from utils.model_utils import init_weight
from utils.data_utils import create_dataset

from model_controller import CNN_Cifar10
from model_controller import CNN_Mnist

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class Server(object):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        self.wandb_on = wandb_on
        self.num_clients = int(num_clients)
        self.selection_ratio = selection_ratio
        self.target_rounds = target_rounds
        self.target_accuracy = target_accuracy
        self.batch_size = batch_size
        
        self.FLgroup = FLgroup
        
        self.len_local_dataset=[]
        self.current_round=0
        
    def setup(self, dataset, iid, split, system_heterogeneity):
        
        # global model초기화    
        printLog(f"PS >> global model을 초기화 합니다.")
        if(dataset=="CIFAR10"):
            self.model_controller = CNN_Cifar10
        elif(dataset=="MNIST"):
            self.model_controller = CNN_Mnist

        self.model = self.model_controller.Model()
        init_weight(self.model)
        
        # local model을 받기 위한 버퍼 생성
        printLog(f"PS >> 빈 local model을 생성합니다.")
        self.flatten_client_models = {}
        
        for idx in range(1,self.num_clients+1):
            self.flatten_client_models[idx] = TensorBuffer(list(self.model.state_dict().values()))

        # train 데이터 분할
        printLog(f"PS >> 데이터셋을 다운받습니다.")
        train_datasets, self.test_data = create_dataset(self.num_clients, dataset, iid, split)

        dist.barrier()

        self.send_local_train_dataset_to_clients(train_datasets)

        self.send_omp_num_threads_to_clients(system_heterogeneity)

    def send_omp_num_threads_to_clients(self, system_heterogeneity):
        if system_heterogeneity>0:
            omp_num_threads_lists = omp_num_threads_per_clients(self.num_clients, system_heterogeneity)
            for idx, threads in enumerate(omp_num_threads_lists):
                dist.send(tensor=torch.tensor([float(threads)]), dst=idx+1)
        elif system_heterogeneity==0:
            for idx in range(self.num_clients): # 모두 thread 2개를 사용해서 학습 진행
                dist.send(tensor=torch.tensor([float(2)]), dst=idx+1)
        dist.barrier()

    def send_local_train_dataset_to_clients(self, train_datasets):
        printLog(f"PS >> 클라이언트들에게 데이터셋을 분할합니다.")
        self.len_local_dataset.append(-1)
        for idx, dataset in enumerate(train_datasets):
            if idx==self.num_clients:
                break
            len_dataset = len(dataset)
            self.len_local_dataset.append(len_dataset)
            for tensor in dataset.tensors:
                tensor_size = torch.tensor(tensor.size()).type(torch.FloatTensor)
                dist.send(tensor=tensor_size, dst=idx+1)
                if(tensor.dtype==torch.int64):
                    tensor=tensor.type(torch.FloatTensor)
                dist.send(tensor=tensor.contiguous(), dst=idx+1)

        dist.barrier()

    def send_num_local_epoch_to_clients(self):
        printLog(f"PS >> 클라이언트들의 local epoch 수를 지정합니다.")
        clients_local_epoch=set_num_local_epoch_by_random(self.num_clients, 5, 15)
        for idx, e in enumerate(clients_local_epoch):
            dist.send(tensor=torch.tensor([float(e)]), dst=idx+1)
        dist.barrier()
        
    def send_global_model_to_clients(self, selected_client_idx):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        for idx in selected_client_idx:
            dist.send(tensor=flatten_model.buffer, dst=idx)

    def wait_local_update_of_selected_clients(self, currentRoundGroup, selected_clients_idx):
        reqs=[]

        for idx in selected_clients_idx:
            req=dist.irecv(tensor=torch.zeros(1))
            reqs.append(req)

        for req in reqs:
            req.wait()
            printLog(f"Server >> CLIENT {req.source_rank()}가 최소 할당량을 완료함")
        
        dist.broadcast(tensor=torch.tensor(float(1)), src=0, async_op=True, group=currentRoundGroup)

    def receive_local_model_from_selected_clients(self, selected_client_idx):
        reqs=[]
        for idx in selected_client_idx:
            req=dist.irecv(tensor=self.flatten_client_models[idx].buffer)
            reqs.append(req)
        for req in reqs:
            req.wait()
    
    def evaluate(self):
        self.model.eval()
	
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.test_data, self.batch_size)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self.model(data)
                test_loss = test_loss + loss_function(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)

                correct = correct + \
                    predicted.eq(labels.view_as(predicted)).sum().item()


        test_loss = test_loss / len(dataloader)
        test_accuracy = correct / len(self.test_data)

        return test_accuracy, test_loss

    def average_aggregation(self, selected_client_idx, coefficient):

        printLog("PS >> global aggregation을 진행합니다.")
        averaged_weights = OrderedDict()
        
        for idx, client_idx in enumerate(selected_client_idx):
            model = self.model_controller.Model()
            model_state_dict=model.state_dict()
            self.flatten_client_models[client_idx].unpack(model_state_dict.values())

            local_weights = model_state_dict
            for key in model_state_dict.keys():
                if idx==0:
                    averaged_weights[key] = coefficient[client_idx] * local_weights[key]
                else:
                    averaged_weights[key] += coefficient[client_idx] * local_weights[key]

        self.model.load_state_dict(averaged_weights)
