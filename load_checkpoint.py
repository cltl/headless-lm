import torch
from engine.tasks.pretraining.mlm_headless import MlmHeadlessPretraining
import sys

import torch.nn as nn
import torch.nn.parallel.distributed as ddp


class DummyDDP(nn.Module):
    def __init__(self, module=None, *args, **kwargs):
        super().__init__()
        self.module = module  # store safely, no delegation

    # IMPORTANT: no __getattr__ override!

    def forward(self, *args, **kwargs):
        if self.module is not None:
            return self.module(*args, **kwargs)
        return None


# override
ddp.DistributedDataParallel = DummyDDP


ckpt_path = sys.argv[1]
pt_output = sys.argv[2]

checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
state_dict = checkpoint["state_dict"]
torch.save({"state_dict": state_dict}, pt_output)
