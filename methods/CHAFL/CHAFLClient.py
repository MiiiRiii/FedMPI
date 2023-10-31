from methods.FedAvg import FedAvgClient

import time
import torch
import torch.distributed as dist

from utils.utils import printLog

from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.nn import CrossEntropyLoss

class CHAFLClient(FedAvgClient.FedAvgClient):
    def doOneLocalEpoch(self, dataloader, optimizer, loss_function):
        for data, labels in dataloader:
            optimizer.zero_grad()
            outputs = self.model.forward(data)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

    def train(self, currentRoundGroup):
        printLog(f"CLIENT {self.id}", "로컬 학습을 시작합니다.")
        
        localEpoch = self.local_epoch-1

        start = time.time()

        self.model.train()
        optimizer = SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_function = CrossEntropyLoss()
        dataloader = DataLoader(self.dataset, self.batch_size, shuffle=True)

        for e in range(localEpoch):
            self.doOneLocalEpoch(dataloader, optimizer, loss_function)
            printLog(f"CLIENT {self.id}", f"{e+1} epoch을 수행했습니다.")    

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
            localEpoch+=1    
            printLog(f"CLIENT {self.id}", f"{localEpoch} epoch을 수행했습니다.")  

        self.total_train_time += time.time()-start
        return localEpoch