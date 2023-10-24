from methods.FedAvg import FedAvgClient

class SemiAsynchronousClient(FedAvgClient.FedAvgClient):
    def __init__(self, num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup):
        super().__init__(num_selected_clients, batch_size, local_epoch, lr, dataset, FLgroup)
        self.num_of_selected=0
        self.model_version=0

    def increase_model_version(self):
        self.model_version += 1