import torch

# Since I've provisioned a VM with 2 GPUs I know need to support training on 2 GPUs, can't just use cuda:0 anymore
class ActiveGPU():
    def __init__(self, available_gpus, default_gpu=0):
        self.available_gpus = available_gpus
        self.gpu_in_use = default_gpu

    def set_gpu(self, gpu_num):
        if gpu_num not in range(len(available_gpus)):
            raise ValueError('Invalid gpu_num given {}, must be in range [0,{}].'.format(gpu_num, len(available_gpus)))

        self.gpu_in_use = gpu_num

    def set_cpu_only(self):
        self.available_gpus = ['cpu']

    def get_gpu_device(self):
        return torch.device(self.available_gpus[self.gpu_in_use])

#available_gpus = ['cuda:0', 'cuda:1']
available_gpus = ['cuda:0']
gpu_in_use = 0
GPU = ActiveGPU(available_gpus, gpu_in_use)

def set_cpu_only():
    GPU.set_cpu_only()

def get_gpu_device():
    return GPU.get_gpu_device()

def set_gpu(gpu_num):
    GPU.set_gpu(gpu_num)