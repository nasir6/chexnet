import os
import numpy as np
import time
import sys
import datetime

from ChexnetTrainer import ChexnetTrainer
from arguments import parse_args
#-------------------------------------------------------------------------------- 

def main (args):

    for arg in vars(args): print(f"{datetime.datetime.now()} --- \t  {arg}: {getattr(args, arg)}")

    try:
        os.makedirs(args.save_dir)
    except OSError:
        pass    
    chexnetTrainer = ChexnetTrainer(args)
    
    if args.test_only:
        assert args.checkpoint is not None, 'no checkpoint file provided to test the model!!!'
        print (f'{datetime.datetime.now()} --- \t Testing the trained model')
        chexnetTrainer.test()
        return
    
    print (f'{datetime.datetime.now()} --- \t Training NN architecture = {args.architecture}')
    chexnetTrainer.train()
    print (f'{datetime.datetime.now()} --- \t Testing the trained model')
    chexnetTrainer.test()


if __name__ == '__main__':
    args = parse_args()
    main(args)





