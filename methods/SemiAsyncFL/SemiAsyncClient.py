from FedAvg import FedAvgClient

class SemiAsynchronousClient(FedAvgClient):
    def __init__(self):
        self.num_of_selected=0
        self.model_version=0

    def increase_model_version(self):
        self.model_version += 1