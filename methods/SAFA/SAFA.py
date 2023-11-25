from utils.utils import *

import wandb
import time
import threading

class SAFA(object):
    def __init__(self):
        None
    
    def runClient(self, Client):
        is_ongoing_local_update_flag = threading.Event() # 로컬 업데이트가 현재 진행중인지
        is_ongoing_local_update_flag.clear()

        terminate_FL_flag = threading.Event()
        terminate_FL_flag.clear()

        listen_global_model = threading.Thread(target=Client.receive_global_model_from_server, args=(is_ongoing_local_update_flag,terminate_FL_flag), daemon=True)
        listen_global_model.start()
        while True:
            if terminate_FL_flag.is_set():
                printLog(f"CLIENT {Client.id}", "FL 프로세스를 종료합니다.")
                break
            if is_ongoing_local_update_flag.is_set():
                is_terminate_FL = Client.train(terminate_FL_flag)
                
                if terminate_FL_flag.is_set() or is_terminate_FL == 1:
                    printLog(f"CLIENT {Client.id}", "FL 프로세스를 종료합니다.")
                    break
                Client.send_local_model_to_server()
                is_ongoing_local_update_flag.clear()

        dist.barrier()
            
    def runServer(self, Server):
        current_FL_start = time.time()
        clients_idx = [idx for idx in range(1,Server.num_clients+1)]
        coefficient = Server.calculate_coefficient(clients_idx)

        P = clients_idx
        Server.send_global_model_to_clients(clients_idx)
        # FL 프로세스 시작
        while True:
            
            current_round_start=time.time()

            Server.current_round+=1

            P, Q = Server.receive_local_model_from_any_clients()

            P, Q = Server.CFCFM(P, Q)

            Server.pre_aggregation_cache_update(P)   

            coefficient = Server.calculate_coefficient(P)         

            Server.average_aggregation(P, coefficient)

            Server.post_aggregation_cache_update(Q)

            Server.send_global_model_to_clients(P,Q)

            global_acc, global_loss = Server.evaluate()



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