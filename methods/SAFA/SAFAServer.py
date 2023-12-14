from methods.FedAvg import FedAvgServer
from utils.model_utils import TensorBuffer
from utils.utils import printLog

import torch.distributed as dist
import torch
import copy
import threading
import gc
import time

from collections import OrderedDict
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

class SAFAServer(FedAvgServer.FedAvgServer):
    def __init__(self, num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup):
        super().__init__(num_clients, selection_ratio, batch_size, target_rounds, target_accuracy, wandb_on, FLgroup)
        self.local_model_version = [-1 for idx in range(0,self.num_clients+1)]

        self.picked_history_per_client={}
        for idx in range(1, self.num_clients+1):
            self.picked_history_per_client[idx]=-1

        self.T_lim = 5000 # WISE
        # self.T_lim = 
        self.Quota=int(self.num_clients*self.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수
        self.flatten_client_models={} #로컬 모델 받는용도
        self.cache={} #aggregation에 사용하는 로컬 모델

        self.idle_clients=[]

        self.lag_tolerance=5

    def setup(self, dataset, iid, split, cluster_type, comm_hetero):
        super().setup(dataset, iid, split, cluster_type, comm_hetero)

        profiling_group = dist.new_group([idx for idx in range(0,self.Quota+1)])
        
        T_dist=time.time()
        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, src=0, group=profiling_group)
        T_dist = time.time() - T_dist

        dist.broadcast(tensor=TensorBuffer(list(self.model.state_dict().values())).buffer, src=0, group=self.FLgroup)

        T_train_k=torch.zeros(1)

        self.clients_est_round_T_train={}

        for idx in range(1, self.num_clients+1):
            req = dist.irecv(tensor=T_train_k)
            req.wait()
            self.clients_est_round_T_train[req.source_rank()] = T_train_k
        
        T_max_train_k = max(self.clients_est_round_T_train.values())

        self.T = min(self.T_lim, T_dist + T_max_train_k)

        printLog(f"SERVER", f"한 라운드의 deadline은 {self.T}입니다.")

        self.cross_rounders=[]
        for c_id in range(1, self.num_clients):
            if self.clients_est_round_T_train[c_id] > self.T:
                self.cross_rounders.append(c_id)
        

        for idx in range(1, self.num_clients+1):
            self.cache[idx] = copy.deepcopy(self.model.state_dict())
    
    def sort_ids_by_perf_desc(self, id_list):
        cp_map={}
        for id in id_list:
            cp_map[id] = self.clients_est_round_T_train[id]

        sorted_map = sorted(cp_map.items(), key=lambda x: x[1], reverse=True)
        sorted_id_list = [sorted_map[i][0] for i in range(len(id_list))]

        return sorted_id_list
    
    def CFCFM(self, make_ids, last_round_pick_ids):

        picks=[]
        in_time_make_ids = [m_id for m_id in make_ids if m_id not in self.cross_rounders]  # in-time make ids
        high_priority_ids = [h_id for h_id in in_time_make_ids if h_id not in last_round_pick_ids]  # compensatory priority
        low_priority_ids = [l_id for l_id in in_time_make_ids if l_id in last_round_pick_ids]
        printLog("SERVER", f"high_priority_ids: {high_priority_ids}")
        printLog("SERVER", f"low_priority_ids: {low_priority_ids}")
        # case 0: clients finishing in time not enough for fraction C, just gather them all
        if len(in_time_make_ids) <= self.Quota:  # if not enough well-progress clients to meet the quota
            return copy.deepcopy(in_time_make_ids)
        # case 1: # of priority ids > quota
        if len(high_priority_ids) >= self.Quota:
            sorted_priority_ids = self.sort_ids_by_perf_desc(high_priority_ids)
            picks = sorted_priority_ids[0:int(self.Quota)]
        # case 2: # of priority ids <= quota
        # the rest are picked by order of performance ("FCFM"), lowest batch overhead first
        else:
            picks += high_priority_ids  # they have priority
            # FCFM
            sorted_low_priority_ids =self. sort_ids_by_perf_desc(low_priority_ids)
            for i in range(min(self.Quota - len(picks), len(sorted_low_priority_ids))):
                picks.append(sorted_low_priority_ids[i])

        return picks


    def version_filter(self, ids, lag_tolerance):
        base_v = self.current_round-1
        good_ids=[]
        deprecated_ids=[]
        for id in ids:
            if base_v - self.local_model_version[id] <= lag_tolerance:
                good_ids.append(id)
            else:
                deprecated_ids.append(id)
        return good_ids, deprecated_ids
    
    def average_aggregation(self, ids, coefficient):
        printLog("SERVER", "global aggregation을 진행합니다.")
        """
        averaged_weights = OrderedDict()
        
        for idx, client_idx in enumerate(ids):
            model = self.model_controller.Model()
            model.load_state_dict(self.cache[client_idx])
            for key in self.cache[client_idx].keys():
                if idx==0:
                    averaged_weights[key] = coefficient[client_idx] * self.cache[client_idx][key]
                else:
                    averaged_weights[key] += coefficient[client_idx] * self.cache[client_idx][key]

        self.model.load_state_dict(averaged_weights)
        """
        global_model_params = self.model.state_dict()
        for pname, param in global_model_params.items():
            global_model_params[pname]=0.0

        for id in ids:
            if id==-1:
                continue
            for pname, param in self.cache[id].items():
                global_model_params[pname] += param.data * coefficient[id]

        self.model.load_state_dict(global_model_params)

    def update_cloud_cache(self, ids):
        model = self.model_controller.Model()
        model_state_dict = model.state_dict()
        for id in ids:
            self.flatten_client_models[id].unpack(model_state_dict.values())
            self.cache[id] = copy.deepcopy(model_state_dict)

    
    def update_cloud_cache_deprecated(self, ids):
        for id in ids:
            self.cache[id] = copy.deepcopy(self.model.state_dict())


    def update_version(self, ids, version):
        for id in ids:
            self.local_model_version[id] = version

    def send_global_model_to_clients(self, ids):
        flatten_model = TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = flatten_model.buffer.tolist()
        global_model_info.append(self.current_round)
        global_model_info = torch.tensor(global_model_info)

        for id in ids:
            dist.send(tensor=flatten_model.buffer, dst=id)
            
    def terminate_FL(self):
        temp_global_model=TensorBuffer(list(self.model.state_dict().values()))
        global_model_info = torch.zeros(len(temp_global_model.buffer)+1)
        global_model_info[-1] = -1
        for idx in range(1, self.num_clients+1):
            dist.send(tensor=global_model_info, dst=idx)



