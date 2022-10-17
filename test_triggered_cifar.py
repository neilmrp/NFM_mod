import numpy as np
import torch
import os
import argparse
import timeit

from torch import nn
from torch.autograd import Variable

from src.get_data import getData, getPoisonedTestSet
import src.cifar_models
from src.noisy_mixup import mixup_criterion
from src.tools import validate, lr_scheduler

torch.cuda.empty_cache()

import sys
np.set_printoptions(threshold=sys.maxsize)



#==============================================================================
# Training settings
#==============================================================================

parser = argparse.ArgumentParser(description='CIFAR10 Example')
#
parser.add_argument('--name', type=str, default='cifar10', metavar='N', help='dataset')
#
parser.add_argument('--test_batch_size', type=int, default=512, metavar='N', help='input batch size for testing (default: 1000)')
#
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 0)')
#
parser.add_argument('--add_trigger', type=str, default='None', metavar='T', help='add trigger during training')
#
args = parser.parse_args()

#==============================================================================
# set random seed to reproduce the work
#==============================================================================
def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True

seed_everything(args.seed)

#==============================================================================
# get device
#==============================================================================
def get_device():
    """Get a gpu if available."""
    if torch.cuda.device_count()>0:
        device = torch.device('cuda')
        print("Connected to a GPU")
    else:
        print("Using the CPU")
        device = torch.device('cpu')
    return device

device = get_device()



train_loader, test_loader = getData(name=args.name, test_bs=args.test_batch_size, trigger=args.add_trigger)
print("Test Dataset Size: ", len(test_loader))
poisoned_test_loader = getPoisonedTestSet(name=args.name, test_bs=args.test_batch_size, trigger=args.add_trigger)
print("Poisoned Test Dataset Size: ", len(poisoned_test_loader))

MODELS_DIRECTORY = '../' + args.name + '_models/'
directory = os.fsencode(MODELS_DIRECTORY)
    
for file in os.listdir(directory):
     filename = os.fsdecode(file)
     if filename.endswith(".pt"):
        model = torch.load(MODELS_DIRECTORY + filename)
        print(filename)

        test_accuracy = validate(test_loader, model, None)
        print('test acc.: %.3f' % test_accuracy)

        poisoned_test_accuracy = validate(poisoned_test_loader, model, None)
        print('poisoned test acc.: %.3f' % poisoned_test_accuracy)


