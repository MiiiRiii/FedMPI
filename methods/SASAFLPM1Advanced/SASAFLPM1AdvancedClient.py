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
from collections import OrderedDict

class SASAFLPM1AdvancedClient(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.local_model_version=0
        self.last_global_loss=100.
        self.current_local_epoch = self.local_epoch


        self.replace_global_model_during_local_update = threading.Event() # 학습 중간에 글로벌 모델로 교체하는지 확인하는 용도
        self.replace_global_model_during_local_update.clear()

        self.interpolate_global_model = threading.Event()
        self.interpolate_global_model.clear()

    def receive_global_model_from_server(self, is_ongoing_local_update_flag, terminate_FL_flag):
        
        self.received_global_model = self.model_controller.Model() # 학습 중간에 받은 글로벌 모델을 담아두는 용도 (바로 self.model에 적용하지 않는 이유: 자원 동시에 접근될 수도 있어서)

        model_state_dict = self.model.state_dict()

        flatten_model = TensorBuffer(list(model_state_dict.values()))
        self.global_model_info = torch.zeros(len(flatten_model.buffer)+4)

        while True:
            dist.recv(tensor=self.global_model_info, src=0) #global_model_info=[flatten_model.buffer, 글로벌 모델 버전(라운드), global loss, 평균 staleness, 선택 여부]
            
            flatten_model.buffer = self.global_model_info[:-4]
            flatten_model.unpack(model_state_dict.values())
            am_i_picked = self.global_model_info[-1].item()
            average_staleness = self.global_model_info[-2].item()
            global_loss = self.global_model_info[-3].item()
            global_model_version = int(self.global_model_info[-4].item())

            if global_model_version == -1: #FL 프로세스 종료
                terminate_FL_flag.set()
                break
            
            elif is_ongoing_local_update_flag.is_set() and global_model_version>=2: # 로컬 학습 중에 글로벌 모델을 받는 경우
                

                """
                if global_model_version-self.local_model_version>average_staleness:
                    printLog(f"CLIENT {self.id}", f"평균 staleness 보다 높으므로 최신 글로벌 모델을 받습니다.")
                    self.received_global_model.load_state_dict(model_state_dict)
                    self.replace_global_model_during_local_update.set()
                """
                if global_loss < self.last_global_loss or global_model_version-self.local_model_version>average_staleness: # 새로운 글로벌 모델이 더 퀄리티가 좋거나 stalness가 심한 경우
                    printLog(f"CLIENT {self.id}", f"gl: {global_loss}, gl^r-si: {self.last_global_loss} 혹은 평균 staleness 보다 높으므로 최신 글로벌 모델을 받습니다.")

                    self.received_global_model.load_state_dict(model_state_dict)
                    self.replace_global_model_during_local_update.set()
                else: # 현재 학습 중인 모델이 더 퀄리티가 좋은 경우
                    printLog(f"CLIENT {self.id}", f"gl: {global_loss}, gl^r-si: {self.last_global_loss}이므로 최신 글로벌 모델을 받지 않고 로컬 업데이트를 이어갑니다.")
                    continue
                

            elif am_i_picked==1: # 처음 라운드이거나 정상적으로 로컬 모델을 업로드 한 후 글로벌 모델을 기다리고 있는 경우
                printLog(f"CLIENT {self.id}", f"처음 라운드이거나 정상적으로 로컬 모델을 업로드 했기 때문에 글로벌 모델을 받습니다.")
                
                self.model.load_state_dict(model_state_dict)

                self.local_model_version = global_model_version
                self.last_global_loss = global_loss
                is_ongoing_local_update_flag.set() # 로컬 업데이트 이제 시작해야 하니까 flag set 해줌

        printLog(f"CLIENT {self.id}", "백그라운드 스레드 종료")

    def train(self, terminate_flag):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
            
        start=time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        epoch_train_loss = 0.0
        e=0
        local_coefficient = 0.9

        while e < self.current_local_epoch:
            if self.replace_global_model_during_local_update.is_set(): # 학습 중간에 글로벌 모델로 교체
                e=0
                epoch_train_loss = 0.0
                self.current_local_epoch -= 1
                
                self.local_model_version = int(self.global_model_info[-4].item())
                self.last_global_loss = self.global_model_info[-3].item()
                self.model = copy.deepcopy(self.received_global_model)
                
                self.replace_global_model_during_local_update.clear()
                continue
            """

            if self.interpolate_global_model.is_set():
                self.current_local_epoch -= e
                e=0
                epoch_train_loss = 0.0

                self.last_global_loss = self.global_model_info[-3].item()

                global_model_state_dict = self.received_global_model.state_dict()
                local_model_state_dict = self.model.state_dict()

                interpolated_weigths = OrderedDict()

                for key in global_model_state_dict.keys():
                    interpolated_weigths[key] = (local_coefficient * local_model_state_dict[key]) + (1-local_coefficient)*global_model_state_dict[key]
                #for key in global_model_state_dict.keys():
                    #interpolated_weigths[key] += (1-local_coefficient) * global_model_state_dict[key]
            
                self.model.load_state_dict(interpolated_weigths)

                continue
            """     
            

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

            if terminate_flag.is_set():
                printLog(f"CLIENT {self.id}", f"학습 도중에 FL 프로세스가 종료되어 로컬 학습을 멈춥니다.")
                utility = -1
                break
            
        self.total_train_time += time.time()-start

        if not terminate_flag.is_set() :
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