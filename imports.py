from datasets import load_dataset
from transformers import GPT2TokenizerFast
from tqdm import tqdm
import os
import csv
import json
import math
import random
import time
import gc
import shutil
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from accelerate import Accelerator, notebook_launcher, DistributedDataParallelKwargs
