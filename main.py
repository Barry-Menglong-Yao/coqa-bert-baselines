import argparse
from model_handler import ModelHandler
from utils.download import download_model
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--trainset', type = str, default = 'data/coqa.train.json', help = 'training dataset file')
parser.add_argument('--devset', type = str, default = 'data/coqa.dev.json', help = 'development dataset file')
parser.add_argument('--model_name', type = str, default = 'BERT', help = '[BERT|RoBERTa|DistilBERT|SpanBERT]')
parser.add_argument('--model_path', type = str, default = None, help = 'path to pretrained model')

parser.add_argument('--cuda', type = str2bool, default = True, help = 'use gpu or not')
parser.add_argument('--debug', type = str2bool, default = True)

parser.add_argument('--n_history', type = int, default = 2, help = 'number of previous question to use as previous context')
parser.add_argument('--batch-size', type = int, default = 2)
parser.add_argument('--shuffle', type = str2bool, default = True)
parser.add_argument('--max_epochs', type = int, default = 5)
parser.add_argument('--lr', type = float, default = 2e-4)
parser.add_argument('--grad_clip', type = float, default = 1.0)
parser.add_argument('--verbose', type = int, default = 200, help = "print after verbose epochs")
parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                    help="Epsilon for Adam optimizer.")
parser.add_argument('--gpu_list', type = str, default = '1'
, help = 'gpu_list,1,2')

parser.add_argument('--save_state_dir', type = str, default = 'output') #output for data and model
parser.add_argument('--pretrained_dir', type = str, default = 'model/10000')#input for model 
parser.add_argument('--preprocessed_data_dir', type = str, default = '10000_turn_id')  #input for data 
parser.add_argument('--mode', type = str, default = 'train', help = ' preprocess or train or test')
parser.add_argument('--data_set_range', type = str, default = 'TRAIN_DATA'
, help = ' TRAIN_DATA or DEV_DATA ')


args = vars(parser.parse_args())

if args['model_name'] == 'SpanBERT':
    download_model()
    args['model_path'] = 'tmp_' 

 

os.environ["CUDA_VISIBLE_DEVICES"] = args['gpu_list']


import torch
print("gpu available: ",torch.cuda.is_available())
print("args: ",args)

handler = ModelHandler(args)
if args['mode']=='train':
    handler.train()
elif args['mode']=='test':
    handler.test()
