from utils.utils import *

import wandb
import time
import threading

class SemiAsyncPM3(object):
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

        Client.terminate()
        dist.barrier()
            
    def runServer(self, Server):
        current_FL_start = time.time()
        clients_idx = [idx for idx in range(1, Server.num_clients+1)]
        picked_client_idx =clients_idx # 학습 처음엔 모든 클라이언트에게 보냄
        minimum_num_picked_client=int(Server.num_clients*Server.selection_ratio) # 한 라운드 동안 수신할 local model 최대 개수
        current_num_picked_client = minimum_num_picked_client
        listen_local_update = threading.Thread(target=Server.receive_local_model_from_any_clients, args=(), daemon=True)
        listen_local_update.start()
        # FL 프로세스 시작
        while True:
            Server.current_round+=1
            
            current_round_start=time.time()

            Server.send_global_model_to_clients(picked_client_idx)

            picked_client_idx  = Server.wait_until_can_update_global_model(current_num_picked_client)

            Server.refine_received_local_model(picked_client_idx )            

            coefficient = Server.calculate_coefficient(picked_client_idx )
            Server.average_aggregation(picked_client_idx , coefficient)

            global_acc, global_loss = Server.evaluate()

            printLog("SERVER", f"{Server.current_round}번째 글로벌 모델 test_accuracy: {round(global_acc*100,4)}%, test_loss: {round(global_loss,4)}")

            if Server.wandb_on=="True":
                wandb.log({"test_accuracy": round(global_acc*100,4), "test_loss":round(global_loss,4), "runtime_for_one_round":time.time()-current_round_start, "wall_time(m)":(time.time()-current_FL_start)/60})
            
            if global_acc>=Server.target_accuracy:
                printLog("SERVER", f"목표한 정확도에 도달했으며, 수행한 라운드 수는 {Server.current_round}회 입니다.")
                printLog("SERVER", "마무리 중입니다..")
                for idx in clients_idx:
                    dist.send(tensor=torch.tensor(-1).type(torch.FloatTensor), dst=idx) # 종료되었음을 알림
                
                break
            
            elif Server.current_round == Server.target_rounds:
                printLog(f"SERVER", f"목표한 라운드 수에 도달했으며, 최종 정확도는 {round(global_acc*100,4)}% 입니다.")
                printLog(f"SERVER", "마무리 중입니다..")
                for idx in clients_idx:
                    dist.send(tensor=torch.tensor(-1).type(torch.FloatTensor), dst=idx) # 종료되었음을 알림
                break

            current_num_picked_client = Server.get_next_round_minimum_local_model(global_loss, current_num_picked_client, minimum_num_picked_client)
        
        Server.terminate(picked_client_idx)
        printLog(f"SERVER", "FL 프로세스를 종료합니다.")
        dist.barrier()