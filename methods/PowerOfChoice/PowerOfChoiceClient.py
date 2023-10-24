from FedAvg import FedAvgClient

import torch
import torch.distributed as dist

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from utils.data_utils import applyCustomDataset

from utils.data_utils import get_uniform_mini_batch
from utils.data_utils import create_uniform_labels
from utils.data_utils import get_local_datasets_labels_probabilities


class PowerOfChoiceClient(FedAvgClient):
    def __init__():
        None

    def receive_local_train_dataset_from_server(self):
        train_data_shape = torch.zeros(4)
        dist.recv(tensor=train_data_shape, src=0)
        data = torch.zeros(train_data_shape.type(torch.int32).tolist())
        dist.recv(tensor=data, src=0)
        
        label_shape = torch.zeros(1)
        dist.recv(tensor=label_shape, src=0)
        label = torch.zeros(label_shape.type(torch.int32).tolist())
        dist.recv(tensor=label, src=0)
        label = label.type(torch.int64)
        
        self.dataset = applyCustomDataset(self.dataset_name, data, label)

        self.unique_labels, self.labels_probabilities = get_local_datasets_labels_probabilities(self.dataset)
        self.num_iteration = len(self.dataset)/self.batch_size
        
        dist.barrier()

    def evaluate(self, method="None"):
        if method == "cpow_d":
            uniform_random_labels = create_uniform_labels(list(self.unique_labels), self.labels_probabilities, self.batch_size)
            uniform_mini_batch = get_uniform_mini_batch(self.dataset_name, self.dataset, uniform_random_labels, self.batch_size)
            dataloader = DataLoader(uniform_mini_batch, self.batch_size)
        
        self.model.eval()
        loss_function = CrossEntropyLoss()
        test_loss, correct = 0, 0

        with torch.no_grad():
            for data, labels in dataloader:
                outputs = self.model(data)
                test_loss = test_loss + loss_function(outputs, labels).item()

                predicted = outputs.argmax(dim=1, keepdim=True)

                correct = correct + \
                    predicted.eq(labels.view_as(predicted)).sum().item()


        test_loss = test_loss / len(dataloader)

        return test_loss