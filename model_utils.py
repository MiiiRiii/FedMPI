import torch
import torch.nn.init as init

def init_weight(model):
    """
    Reference:
        https://github.com/DS3Lab/forest-prediction/blob/master/pix2pix/models/networks.py
    """
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            init.xavier_normal_(m.weight.data, gain=1.0)
        if hasattr(m, 'bias') and m.bias is not None:
            init.constant_(m.bias.data, 0.0)

    model.apply(init_func)


def flatten(tensors, shapes=None, use_cuda=False):
    """
        Reference:
        https://github.com/epfml/federated-learning-public-code/blob/master/codes/FedDF-code/pcode/utils/communication.py
    """
    # init and recover the shapes vec.
    pointers = [0]
    if shapes is not None:
        for shape in shapes:
            pointers.append(pointers[-1] + shape[1])
    else:
        for tensor in tensors:
            pointers.append(pointers[-1] + tensor.nelement())

    # flattening.
    vec = torch.empty(
        pointers[-1],
        device=tensors[0].device if tensors[0].is_cuda and use_cuda else "cpu",
    )

    for tensor, start_idx, end_idx in zip(tensors, pointers[:-1], pointers[1:]):
        vec[start_idx:end_idx] = tensor.data.view(-1)
    return vec

class TensorBuffer:
    """
    Packs multiple tensors into one flat buffer for efficient
    intra-worker communication.
    """

    def __init__(self, tensors, use_cuda=False):
        indices = [0]
        for tensor in tensors:
            new_end = indices[-1] + tensor.nelement()
            indices.append(new_end)

        self._start_idx = indices[:-1]
        self._end_idx = indices[1:]
        self._tensors_len = len(tensors)
        self._tensors_sizes = [x.size() for x in tensors]

        self.buffer = flatten(tensors, use_cuda=use_cuda)  # copies

    def __getitem__(self, index):
        return self.buffer[self._start_idx[index] : self._end_idx[index]].view(
            self._tensors_sizes[index]
        )

    def __len__(self):
        return self._tensors_len

    def is_cuda(self):
        return self.buffer.is_cuda

    def nelement(self):
        return self.buffer.nelement()

    def unpack(self, tensors):
        for tensor, entry in zip(tensors, self):
            tensor.data = entry.clone()
