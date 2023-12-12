from utils.utils import *

import wandb
import time
import threading
import copy
import random

def generate_crash_trace(num_clients, num_rounds):
    clients_crash_prob_vec = [0.3 for _ in range(0, num_clients+1)]
    crash_trace=[]
    progress_trace=[]
    for r in range(num_rounds):
        crash_ids=[]
        progress = [1.0 for _ in range(0,num_clients+1)]
        for c_id in range(1, num_clients+1):
            rand=random.random()
            if rand <= clients_crash_prob_vec[c_id]:
                crash_ids.append(c_id)
                progress[c_id] = rand / clients_crash_prob_vec[c_id]
        crash_trace.append(crash_ids)
        progress_trace.append(progress)
    
    return crash_trace, progress_trace




class SAFA(object):
    def __init__(self, lag_tolerance):
        self.lag_tolerance = lag_tolerance
    
    def runClient(self, Client):
        
        terminate_FL_flag = threading.Event()
        terminate_FL_flag.clear()

        listen_global_model = threading.Thread(target=Client.receive_global_model_from_server, args=(terminate_FL_flag,), daemon=True)
        listen_global_model.start()

        while True:
            if terminate_FL_flag.is_set():
                break
            make_ids_list=torch.tensor([0 for _ in range(0,Client.num_clients)])
            dist.broadcast(tensor=make_ids_list, src=0, group=Client.FLgroup)
            if terminate_FL_flag.is_set():
                break
            if Client.id in make_ids_list:
                Client.train()
                Client.send_local_model_to_server()


        dist.barrier()
            
    def runServer(self, Server):
        current_FL_start = time.time()
        clients_idx = [idx for idx in range(1,Server.num_clients+1)]
        coefficient = Server.calculate_coefficient(clients_idx)

        P = clients_idx
        Server.lag_tolerance = self.lag_tolerance

        crash_trace, progress_trace = generate_crash_trace(Server.num_clients, Server.target_rounds)

        picked_ids = []
        # FL 프로세스 시작
        while True:
            
            current_round_start=time.time()

            crash_ids = crash_trace[Server.current_round]
            printLog("SERVER", f"crashed clients: {crash_ids}")
            make_ids = [c_id for c_id in range(1, Server.num_clients+1) if c_id not in crash_ids]
            printLog("SERVER", f"make clients: {make_ids}")

            # compensatory first-come-first-merge selection, last-round picks are considered low priority
            picked_ids = Server.CFCFM(make_ids, picked_ids)
            printLog("SERVER", f"make clients: {make_ids}")
            
            undrafted_ids = [c_id for c_id in make_ids if c_id not in picked_ids]
            printLog("SERVER", f"undrafted clients: {undrafted_ids}")

            # distributing step
            # distribute the global model to the edge in a discriminative manner
            good_ids, deprecated_ids = Server.version_filter(clients_idx, Server.lag_tolerance)
            printLog("SERVER", f"good clients: {good_ids}")
            printLog("SERVER", f"deprecated clients: {deprecated_ids}")

            latest_ids, straggler_ids = Server.version_filter(good_ids, 0)
            printLog("SERVER", f"latest clients: {latest_ids}")

            # case 1: deprecated clients
            Server.send_global_model_to_clients(deprecated_ids)
            Server.update_cloud_cache_deprecated(deprecated_ids)
            Server.update_version(deprecated_ids, Server.current_round-1)
            # case 2: latest clients
            Server.send_global_model_to_clients(latest_ids)
            # case 3: non-deprecated stragglers
            # Moderately straggling clients remain unsync.

            # Local Update
            
            
            if Server.current_round==0:
                make_ids_list = clients_idx
            else:
                make_ids_list = make_ids+[-1 for _ in range(Server.num_clients-len(make_ids))]
            dist.broadcast(tensor=torch.tensor(make_ids_list), src=0, group=Server.FLgroup)
            Server.receive_local_model_from_selected_clients(make_ids_list)



            # Aggregation step
            # discriminative update of cloud cache and aggregate
            # pre-aggregation: update cache from picked clients
            Server.update_cloud_cache(picked_ids)        
            # SAFA aggregation
            Server.average_aggregation(clients_idx, coefficient)
            global_acc, global_loss = Server.evaluate()
            # post=aggregation
            Server.update_cloud_cache(undrafted_ids)

            # update_version
            Server.update_version(make_ids, Server.current_round)

            Server.current_round+=1

            printLog("SERVER", f"{Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})
            
            if global_acc>=Server.target_accuracy:
                printLog("SERVER", f"목표한 정확도에 도달했으며, 수행한 라운드 수는 {Server.current_round}회 입니다.")
                
                break
            
            elif Server.current_round == Server.target_rounds:
                printLog(f"SERVER", f"목표한 라운드 수에 도달했으며, 최종 정확도는 {round(global_acc*100,4)}% 입니다.")
                break
        Server.terminate_FL()
        printLog(f"SERVER", "FL 프로세스를 종료합니다.")
        dist.barrier()