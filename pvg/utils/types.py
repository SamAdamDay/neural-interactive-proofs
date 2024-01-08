import logging

import torch

TorchDevice = torch.device | str | int

LoggingType = logging.Logger | logging.LoggerAdapter