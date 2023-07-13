import torch.distributed as dist
import torch
import time

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


from utils import printLog
from data_utils import applyCustomDataset
from model_controller import CNN_Cifar10
from model_controller import CNN_Mnist
from model_utils import TensorBuffer


class Client(object):
    def __init__(self, num_clients, selection_ratio, batch_size, local_epoch, lr, dataset, FLgroup):
        self.id=dist.get_rank()
        self.num_selected_clients = int(num_clients * selection_ratio)
        self.batch_size = batch_size
        self.local_epoch = local_epoch
        self.lr = lr
        self.dataset_name=dataset
        self.total_train_time=0
        self.num_of_selected=0
        
        self.FLgroup = FLgroup
    
    def setup(self):
        printLog(f"CLIENT {self.id} >> 빈 모델을 생성합니다.")
        if(self.dataset_name == "CIFAR10"):
            self.model_controller = CNN_Cifar10
        elif(self.dataset_name == "MNIST"):
            self.model_controller = CNN_Mnist
        
        self.model = self.model_controller.Model()

        dist.barrier()
        
        # train dataset receive
        self.receive_local_train_dataset_from_server()

        dist.barrier()


    
    def receive_local_train_dataset_from_server(self):

        train_data_shape = torch.zeros(4)
        dist.recv(tensor=train_data_shape, src=0)
        data = torch.zeros(train_data_shape.type(torch.int32).tolist())
        dist.recv(tensor=data, src=0)
        
        label_shape = torch.zeros(1)
        dist.recv(tensor=label_shape, src=0)
        label = torch.zeros(label_shape.type(torch.int32).tolist())
        dist.recv(tensor=label, src=0)
        label = label.type(torch.int64)
        
        self.dataset = applyCustomDataset(self.dataset_name, data, label)
        
        
    def train(self):
        printLog(f"CLIENT {self.id} >> 로컬 학습을 시작합니다.")
        self.num_of_selected += 1
        self.model.train()

        start=time.time()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        for e in range(self.local_epoch):
            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
            printLog(f"CLIENT {self.id} >> {e+1} epoch을 수행했습니다.")
        
        self.total_train_time += time.time()-start
    
    def receive_global_model_from_server(self):
        model_state_dict = self.model.state_dict()
        model_tb = TensorBuffer(list(model_state_dict.values()))
        dist.recv(tensor=model_tb.buffer, src=0)
        model_tb.unpack(model_state_dict.values())
        self.model.load_state_dict(model_state_dict)

    def send_local_model_to_server(self):
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)

    
    def start(self):
        
        while True:
            selected=False
            selected_clients=torch.zeros(self.num_selected_clients).type(torch.int64)
            dist.broadcast(tensor=selected_clients, src=0, group=self.FLgroup)
            
            for idx in selected_clients:
                if idx == self.id:
                    selected=True
                    break

            if(selected):
                self.receive_global_model_from_server()
                self.train()
                printLog(f"CLIENT {self.id} >> 평균 학습 소요 시간: {self.total_train_time/self.num_of_selected}")
                self.send_local_model_to_server()

            dist.barrier()

            continueFL = torch.zeros(1)
            dist.broadcast(tensor=continueFL, src=0, group=self.FLgroup)
            if(continueFL[0]==0): #FL 종료
                break
            