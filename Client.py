import torch.distributed as dist
import torch
import time

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss


from utils.utils import printLog
from utils.data_utils import applyCustomDataset
from utils.data_utils import get_local_datasets_labels_probabilities
from utils.data_utils import get_uniform_mini_batch
from utils.data_utils import create_uniform_labels

from model_controller import CNN_Cifar10
from model_controller import CNN_Mnist

from utils.model_utils import TensorBuffer


class Client(object):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        self.num_selected_clients=num_selected_clients
        self.id=dist.get_rank()
        self.batch_size = batch_size
        self.local_epoch = local_epoch
        self.lr = lr
        self.dataset_name=dataset
        self.total_train_time=0
        self.num_of_selected=0
        
        self.FLgroup = FLgroup
    
    def setup(self, cluster_type, num_thread):
        printLog(f"CLIENT {self.id} >> 빈 모델을 생성합니다.")
        if(self.dataset_name == "CIFAR10"):
            self.model_controller = CNN_Cifar10
        elif(self.dataset_name == "MNIST"):
            self.model_controller = CNN_Mnist
        elif(self.dataset_name=="FashionMNIST"):
            self.model_controller = CNN_Mnist

        self.model = self.model_controller.Model()

        dist.barrier()
        
        # receive train dataset 
        self.receive_local_train_dataset_from_server()
        
        if cluster_type == "KISTI":
            torch.set_num_threads(int(num_thread))
        else:
            self.receive_omp_num_threads_from_server()
    
    def receive_omp_num_threads_from_server(self):
        tensor=torch.zeros(1)
        dist.recv(tensor=tensor, src=0)
        torch.set_num_threads(int(tensor.item()))
        printLog(f"CLIENT {self.id} >> thread {int(tensor.item())}개를 사용합니다.")
        dist.barrier()

    def receive_num_local_epoch_from_server(self):
        tensor = torch.zeros(1)
        dist.recv(tensor=tensor, src=0)
        self.local_epoch = int(tensor.item())
        printLog(f"CLIENT {self.id} >> local epoch 수는 {self.local_epoch}입니다.")
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

        self.unique_labels, self.labels_probabilities = get_local_datasets_labels_probabilities(self.dataset)
        self.num_iteration = len(self.dataset)/self.batch_size
        dist.barrier()

    def doOneLocalEpoch(self, dataloader, optimizer, loss_function):
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = self.model.forward(data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    def train(self, currentRoundGroup=None):
        printLog(f"CLIENT {self.id} >> 로컬 학습을 시작합니다.")
        self.num_of_selected += 1
        if currentRoundGroup!=None: 
            performedLocalEpoch=self.local_epoch-1
        else:
            performedLocalEpoch=self.local_epoch

        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)
        
        
        for e in range(performedLocalEpoch):
            self.doOneLocalEpoch(dataloader, optimizer, loss_function)
            printLog(f"CLIENT {self.id} >> {e+1} epoch을 수행했습니다.")

        if currentRoundGroup!=None: # If current method is CHAFL
            # Notify completion minimum quota to server
            dist.send(tensor=torch.tensor([float(1)]), dst=0)

            # Receive message asynchronously from server
            req=dist.broadcast(tensor=torch.zeros(1), src=0, async_op=True, group=currentRoundGroup)

            # Do additional local epoch
            continueLocalUpdate=True
            
            while continueLocalUpdate:
                if req.is_completed():
                    continueLocalUpdate=False
                    break
                self.doOneLocalEpoch(dataloader, optimizer, loss_function)
                performedLocalEpoch+=1    
                printLog(f"CLIENT {self.id} >> {performedLocalEpoch} epoch을 수행했습니다.")  
        
        self.total_train_time += time.time()-start
        return performedLocalEpoch
    
    def evaluate(self, method="None"):
        self.model.eval()
	
        loss_function = CrossEntropyLoss()
        if method == "cpow_d":
            uniform_random_labels = create_uniform_labels(list(self.unique_labels), self.labels_probabilities, self.batch_size)
            uniform_mini_batch = get_uniform_mini_batch(self.dataset_name, self.dataset, uniform_random_labels, self.batch_size)
            dataloader = DataLoader(uniform_mini_batch, self.batch_size)
        else:
            dataloader = DataLoader(self.dataset, self.batch_size)

        test_loss, correct = 0, 0
        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self.model(data)
                test_loss = test_loss + loss_function(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)

                correct = correct + \
                    predicted.eq(labels.view_as(predicted)).sum().item()


        test_loss = test_loss / len(dataloader)

        return test_loss
    
    def receive_global_model_from_server(self):
        model_state_dict = self.model.state_dict()
        model_tb = TensorBuffer(list(model_state_dict.values()))
        dist.recv(tensor=model_tb.buffer, src=0)
        model_tb.unpack(model_state_dict.values())
        self.model.load_state_dict(model_state_dict)

    def send_local_model_to_server(self):
        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        dist.send(tensor=flatten_model.buffer, dst=0)