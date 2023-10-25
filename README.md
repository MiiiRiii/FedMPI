Implementation Federated Learning Platform using torch.distributed

How to run:

1. Activate conda on the node where the server and client process will run (using FL.yaml file)
   
2. Enter the following command on the node.
$ torchrun \
--nnodes=(number of nodes to use) \
--nproc_per_node=(number of processes to use per node) \
--node_rank=(Global rank of the node executing this command) \
--rdzv_id=0 \
--rdzv_endpoint=(IP address of the node where the server/master process is located):(Port number to communicate with server/master process) \
--master_addr=(IP address of the node where the server/master process is located) \
--master_port=(Port number to communicate with server/master process) \
Start.py \
--selection_ratio=(ratio of clients selected per round) \
--round=(total number of rounds to perform) \
--batch_size=(batch size) \
--local_epoch=(local epoch) \
--lr=(local learning rate) \
--target_acc=(accuracy targeted by the global model) \
--dataset=(dataset name) \
--iid=(dataset distribution) \
--split=(distribution name that divides the training data) \
--method=(method name to run) \
--d=(number of randomly selected clients that must be set when using the power of choice method; If method name is not pow_d or cpow_d, no need to set this argument) \
--wandb_on=(whether to use wandb) \
--project=(wandb project name; If wandb_on is false, no need to set this argument) \
--entity=(username performing the record wandb; If wandb_on is false, no need to set this argument) \
--group=(group name to which this record belongs; If wandb_on is false, no need to set this argument) \
--name=(record namel; If wandb_on is false, no need to set this argument) \
--repeat=(total number of FL process iterations) \
--cluster_type=(if you want to arbitrarily adjust the number of omp threads for each client, use "WISE". If you want to automatically adjust the number of omp threads, use "KISTI")

Example) 
$ torchrun --nnodes=6 --nproc_per_node=6 --node_rank=5 --rdzv_id=0 --rdzv_endpoint=wise167:29603 --master_addr=wise167 --master_port=29603 FedAvg.py --selection_ratio=0.1 --round=1000 --batch_size=32 --local_epochs=5 --lr=0.001 --target_acc=0.7 --dataset=CIFAR10 --iid=False --split=gaussian --method=FeAvg --wandb_on=False --repeat=1 --cluster_type=WISE

