import logging

import torch

from numpy.dtypes import StringDType

TorchDevice = torch.device | str | int

LoggingType = logging.Logger | logging.LoggerAdapter

NumpyStringDtype = StringDType(na_object=None)
