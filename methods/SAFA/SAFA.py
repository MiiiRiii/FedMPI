from utils.utils import *

import wandb
import time
import threading

class SAFA(object):
    def __init__(self):
        None
    
    def runClient(self, Client):
        while True:
            isTerminate = Client.receive_global_model_from_server()
            if isTerminate == 0 :
                printLog(f"CLIENT{Client.id}", "FL 프로세스를 종료합니다.")

                break
            Client.train()
            Client.send_local_model_to_server()

        dist.barrier()
            
    def runServer(self, Server):
        current_FL_start = time.time()

        # FL 프로세스 시작
        while True:
            Server.current_round+=1
            
            current_round_start=time.time()

            Server.send_global_model_to_clients()

            P, Q = Server.receive_local_model_from_any_clients()

            P, Q = Server.CFCFM(P, Q)

            cache = Server.pre_aggregation_cache_update(P)            

            coefficient = Server.calculate_coefficient(P)
            Server.average_aggregation(P, coefficient)

            Server.post_aggregation_cache_update()

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
        printLog(f"SERVER", "마무리 중입니다..")
        Server.terminate(picked_client_idx)
        printLog(f"SERVER", "FL 프로세스를 종료합니다.")
        dist.barrier()