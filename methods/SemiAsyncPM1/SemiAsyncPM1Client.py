from methods.FedAvg import FedAvgClient
from utils.utils import *
from utils.model_utils import TensorBuffer

import torch.distributed as dist

import torch
import time
import threading
import math
import copy

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class SemiAsyncPM1Client(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
        self.last_global_loss=100.
        self.current_local_epoch = self.local_epoch


        self.receive_global_model_flag = threading.Event() # 학습 중간에 글로벌 모델을 받았는지 확인하는 용도
        self.receive_global_model_flag.clear()



    def receive_global_model_from_server(self, is_ongoing_local_update_flag, terminate_FL_flag):
        
        self.received_global_model = self.model_controller.Model() # 학습 중간에 받은 글로벌 모델을 담아두는 용도 (바로 self.model에 적용하지 않는 이유: 자원 동시에 접근될 수도 있어서)

        model_state_dict = self.model.state_dict()

        flatten_model = TensorBuffer(list(model_state_dict.values()))
        global_model_info = torch.zeros(len(flatten_model.buffer)+3)

        while True:
            dist.recv(tensor=global_model_info, src=0) #global_model_info=[flatten_model.buffer, 글로벌 모델 버전(라운드), global loss]
            
            flatten_model.buffer = global_model_info[:-3]
            flatten_model.unpack(model_state_dict.values())
            am_i_picked = global_model_info[-1].item()

            if int(global_model_info[-3].item()) == -1: #FL 프로세스 종료
                terminate_FL_flag.set()
                break

            elif is_ongoing_local_update_flag.is_set() and int(global_model_info[-3].item())>=2: # 로컬 학습 중에 글로벌 모델을 받는 경우

                if global_model_info[-2].item() < self.last_global_loss: # 새로운 글로벌 모델이 더 퀄리티가 좋은 경우


                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[-2].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받습니다.")

                    self.received_global_model.load_state_dict(model_state_dict)
                    
                    self.local_model_version = int(global_model_info[-3].item())
                    self.last_global_loss = global_model_info[-2].item()

                    self.receive_global_model_flag.set()


                else: # 현재 학습 중인 모델이 더 퀄리티가 좋은 경우
                    printLog(f"CLIENT {self.id}", f"gl: {global_model_info[-2].item()}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받지 않고 로컬 업데이트를 이어갑니다.")
                    continue

            elif am_i_picked==1: # 처음 라운드이거나 정상적으로 로컬 모델을 업로드 한 후 글로벌 모델을 기다리고 있는 경우
                printLog(f"CLIENT {self.id}", f"처음 라운드이거나 정상적으로 로컬 모델을 업로드 했기 때문에 글로벌 모델을 받습니다.")
                
                self.model.load_state_dict(model_state_dict)

                self.local_model_version = int(global_model_info[-3].item())
                self.last_global_loss = global_model_info[-2].item()

                #local_loss = super().evaluate()
                #printLog(f"CLIENT {self.id}", f"local loss: {local_loss}")

                is_ongoing_local_update_flag.set() # 로컬 업데이트 이제 시작해야 하니까 flag set 해줌

        printLog(f"CLIENT {self.id}", "백그라운드 스레드 종료")

    def train(self):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            
        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        epoch_train_loss = 0.0
        e=0

        while e < self.current_local_epoch:
            if self.receive_global_model_flag.is_set(): # 학습 중간에 글로벌 모델을 받았다면 교체
                e=0
                epoch_train_loss = 0.0
                self.current_local_epoch -= 1
                self.receive_global_model_flag.clear()
                self.model = copy.deepcopy(self.received_global_model)
                #local_loss = super().evaluate()
                #printLog(f"CLIENT {self.id}", f"local loss: {local_loss}")
                continue

            for data, labels in dataloader:
                optimizer.zero_grad()
                outputs = self.model.forward(data)
                loss = loss_function(outputs, labels)
                
                if e==0:
                    epoch_train_loss += loss.detach().item()

                loss.backward()
                optimizer.step()
            e+=1

            printLog(f"CLIENT {self.id}", f"{e} epoch을 수행했습니다.")
            
        self.total_train_time += time.time()-start

        #utility = math.sqrt(epoch_train_loss / self.len_local_dataset) * self.len_local_dataset
        utility = epoch_train_loss / len(dataloader)
        printLog(f"CLIENT {self.id}", f"local utility: {utility}")

        return utility
    
    def send_local_model_to_server(self, utility):
        self.current_local_epoch = self.local_epoch

        flatten_model=TensorBuffer(list(self.model.state_dict().values()))
        local_model_info = flatten_model.buffer.tolist()
        local_model_info.append(utility)
        local_model_info.append(self.local_model_version)
        local_model_info = torch.tensor(local_model_info)
        
        dist.send(tensor=local_model_info, dst=0)

    
    
    def terminate(self):
        dist.barrier()
        # 클라이언트 1이 대표로 실행
        if self.id == 1:
            isTerminate = torch.zeros(1)
            dist.recv(tensor = isTerminate, src=0)
            if isTerminate == 0:
                self.send_local_model_to_server(0)
    

