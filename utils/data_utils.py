import json
import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch 
from collections import Counter, defaultdict
import numpy as np
from random import shuffle
import math
import textacy.preprocessing.replace as rep
from tqdm import tqdm
import spacy
import csv
import pickle
import os.path
import utils.timer  as  timer
nlp = spacy.load('en_core_web_sm')
 
TRAIN_COQA_DATASET_VERSION1_FILENAME="temp_data/train/train_coqa_dataset_version1.pt"
DEV_COQA_DATASET_FILENAME="temp_data/dev/dev_coqa_dataset.pt"
TRAIN_TOKENIZER_VERSION1_FILENAME="temp_data/saved/data/40000/train_tokenizer_version1.pt"
DEV_TOKENIZER_VERSION3_FILENAME= "temp_data/dev/dev_tokenizer_version3.pt"
DATA_PATH="temp_data/saved/data/" 


number_of_part1=125000   #25000 for 4 hours. 125000 for all. 3 for test.

def prepare_datasets(config, tokenizer_model):
    tokenizer = tokenizer_model[1].from_pretrained(tokenizer_model[2])
    if config['data_set_range']=='DEV_DATA':
        data_set_range=DATA_SET_RANGE.DEV_DATA 
    else:
        data_set_range=DATA_SET_RANGE.TRAIN_DATA 
    if config['mode']=='train' or  config['mode']=='test':
        preprocess_step=PROCESS_STEP.TRAIN__LOAD_ALL_DATA
    else:
        preprocess_step=PROCESS_STEP.PREPROCESS__SPLIT_DATA_AND_SAVE
    trainloader=None 
    devloader=None 

    if preprocess_step==PROCESS_STEP.PREPROCESS__SPLIT_DATA_AND_SAVE:
        if data_set_range==DATA_SET_RANGE.DEV_DATA: 
            tokenizer=load_tokenizer(config['preprocessed_data_dir'], preprocess_step,data_set_range,tokenizer)
            devset = CoQADataset(config['devset'],preprocess_step)
            devset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,data_set_range)
            devloader = CustomDataLoader(devset, config['batch_size'])
        else:
            trainset = CoQADataset(config['trainset'],preprocess_step)
            trainset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,data_set_range)
            trainloader = CustomDataLoader(trainset, config['batch_size'])
            
    else :
 
        # trainset = CoQADataset(config['trainset'],preprocess_step)
        # trainset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,DATA_SET_RANGE.TRAIN_DATA)
        # trainloader = CustomDataLoader(trainset, config['batch_size'])
        # devset = CoQADataset(config['devset'],preprocess_step)
        # devset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,DATA_SET_RANGE.DEV_DATA )
        # devloader = CustomDataLoader(devset, config['batch_size']) 

         
        # trainset1 = torch.load(TRAIN_COQA_DATASET_VERSION1_FILENAME) 
        # tokenizer1=load_tokenizer(preprocess_step, data_set_range, tokenizer)
        # devset1 = torch.load(DEV_COQA_DATASET_FILENAME) 


        trainset1 = load_data(DATA_PATH+config['preprocessed_data_dir']+"/train_coqa_dataset_version1.pt") 
        trainloader1 = CustomDataLoader(trainset1, config['batch_size'])
        devset1 = load_data(DATA_PATH+config['preprocessed_data_dir']+"/dev_coqa_dataset.pt") 
        devloader1 = CustomDataLoader(devset1, config['batch_size']) 
        tokenizer1=load_tokenizer(config['preprocessed_data_dir'], preprocess_step, data_set_range, tokenizer)
        print('Train Load   {} chunk_examples.'.format(len(trainset1.chunked_examples) ))
        print('Dev Load {} paragraphs, {} examples.'.format(len(devset1.paragraphs), len(devset1.examples)))
    # return trainloader1, devloader1, tokenizer1
    return trainloader1, devloader1, tokenizer1
 

def load_data(filename):
    print("load from ",filename)
    return torch.load(filename)

def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)

def preprocess(text):
    text = ' '.join(text)
    temp_text = rep.replace_currency_symbols(text, replace_with = '_CUR_')
    temp_text = rep.replace_emails(temp_text, replace_with = '_EMAIL_')
    temp_text = rep.replace_emojis(temp_text, replace_with='_EMOJI_')
    temp_text = rep.replace_hashtags(temp_text, replace_with='_TAG_')
    temp_text = rep.replace_numbers(temp_text, replace_with='_NUMBER_')
    temp_text = rep.replace_phone_numbers(temp_text, replace_with = '_PHONE_')
    temp_text = rep.replace_urls(temp_text, replace_with = '_URL_')
    temp_text = rep.replace_user_handles(temp_text, replace_with = '_USER_')

    doc = nlp(temp_text)
    tokens = []
    for t in doc:
        tokens.append(t.text)
    return tokens


from enum import Enum
class PROCESS_STEP(Enum):
    PREPROCESS__SPLIT_DATA_AND_SAVE = 1  #is spliting data into multi parts and save them. will save from the last checkpoint
    TRAIN__LOAD_ALL_DATA = 2   #is in training step. Just load all dataset and cnt=0
    DEBUG_WITH_SMALL_DATA =3

class DATA_SET_RANGE(Enum):
    DEV_DATA=1
    TRAIN_DATA=2
    TRAIN_DATA_SECOND_PART=3


def loadall( filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def save_object(obj, filename):
        pickle.dump(obj, filename) 
class CoQADataset(Dataset):
    """CoQA dataset."""


    def load_from_file(self,filename ):
        if "train" in filename:
            with open('temp_data/train/example.pkl', 'rb') as input:
                self.examples = pickle.load(input)

            with open('temp_data/train/paragraph.pkl', 'rb') as input:
                self.paragraphs = pickle.load(input) 
        else:
            with open('temp_data/dev/dev.json_example.pkl', 'rb') as input:
                self.examples = pickle.load(input)

            with open('temp_data/dev/dev.json_paragraph.pkl', 'rb') as input:
                self.paragraphs = pickle.load(input) 

        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))  

    """
    1,self.paragraphs.append(paragraph)
    2,
    qas['annotated_question']['word'] is conversation history: Q1 A1 Q2 A2 ... current_Q
    self.examples.append(qas)
    3,self.vocab[w] += 1
    """
    def __init__(self, filename,preprocess_step):
        #timer = Timer('Load %s' % filename)
        self.filename = filename
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        self.chunked_examples = []        
        # dataset = read_json(filename)
        # print("Read started.   ", len(dataset['data']))

        #'''
        if preprocess_step==PROCESS_STEP.DEBUG_WITH_SMALL_DATA :
            self.do_original_init_process(filename)
        else:
            self.load_from_file(filename)

        

    def do_original_init_process(self,filename):
        #timer = Timer('Load %s' % filename)
        self.filename = filename
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
         
        dataset = read_json(filename)
        cnt=0
        for paragraph in tqdm(dataset['data']):
            if cnt>number_of_part1:
                break
            else:
                cnt+=1
            #print(paragraph)
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) #if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        q1 = preprocess(q)
                        a1 = preprocess(a)
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q1)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a1)
                temp.append('<Q>')
                temp.extend(qas['annotated_question']['word'])
                history.append((qas['annotated_question']['word'], qas['annotated_answer']['word']))
                qas['annotated_question']['word'] = temp
                self.examples.append(qas)
                question_lens.append(len(qas['annotated_question']['word']))
                paragraph_lens.append(len(paragraph['annotated_context']['word']))
                for w in qas['annotated_question']['word']:
                    self.vocab[w] += 1
                for w in paragraph['annotated_context']['word']:
                    self.vocab[w] += 1
                for w in qas['annotated_answer']['word']:
                    self.vocab[w] += 1
            self.paragraphs.append(paragraph)
        print('Load {} paragraphs, {} examples.'.format(len(self.paragraphs), len(self.examples)))
        print('Paragraph length: avg = %.1f, max = %d' % (np.average(paragraph_lens), np.max(paragraph_lens)))
        print('Question length: avg = %.1f, max = %d' % (np.average(question_lens), np.max(question_lens)))

        
    


    def load_cnt(self):
        if data_set_range !=DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            num=0
        else:
            file_path = 'temp_data/count.txt'
            num = 0
            if os.path.isfile(file_path)    :
                with open(file_path, newline='') as f:
                    rows = list(csv.reader(f))
                for row in rows:
                    #print(type(row), row)
                    num = int(row[0])
                print("Mx count =======     ", num)

        cnt = num
        return cnt

    def save_cnt(self,cnt):
        file_path = 'temp_data/count.txt'
        f = open(file_path, "w")
            #if(cnt>=num):
        f.write(str(cnt))
        f.close()

    

    
     
 

    ##
    # generate input_tokens for BERT and save in self.chunked_examples  (can use minibatch)
    #  tokenizer.add_tokens()
    def chunk_paragraphs(self, tokenizer, model_name ,preprocess_step,data_set_range):
        #  when preprocess the second 50000 examples, we need firstly load the first 50000 token into tokenizer. Otherwise, the token_id of a word may change. For example, from 50010 to 10.
        # tokenizer=  load_tokenizer(preprocess_step,data_set_range,tokenizer)
        # print("Chunk paragrapsh begin.      tokenizer number: {} ".format(len(tokenizer))  ) 
        # cnt = self.load_cnt(data_set_range)
        c_unknown = 0
        c_known = 0
        dis = 0


        
        if data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            start_idx=number_of_part1
        else:
            start_idx=0
         
        timer1=timer.Timer()
        for i, ex in tqdm(enumerate(self.examples[start_idx::])):
            total=min([len(self.examples[start_idx::]),number_of_part1])

            # if i >total:
            #     break
            if (i+1)%5000==0:
                self.save_tokenizer(tokenizer,data_set_range)
                self.save_coqa_dataset(data_set_range) #chunked_examples
                print(timer1.remains(total,i))
                
            
              
            question_length = len(ex['annotated_question']['word'])
            if question_length > 350:  
                continue
            doc_length_available = 512 - question_length - 3
            if model_name == 'RoBERTa':
                doc_length_available = doc_length_available - 3
            
            paragraph = self.paragraphs[ex['paragraph_id']]['annotated_context']['word']
            paragraph = preprocess(paragraph)
            if model_name != 'RoBERTa' and model_name != 'SpanBERT':
                paragraph = [p.lower() for p in paragraph]
            paragraph_length = len(paragraph)
            start_offset = 0
            doc_spans = []
            while start_offset < paragraph_length:
                length = paragraph_length - start_offset
                if length > doc_length_available:
                    length = doc_length_available - 1
                    doc_spans.append([start_offset, length, 1])
                else:
                    doc_spans.append([start_offset, length, 0])
                if start_offset + length == paragraph_length:
                    break
                start_offset += length
            for spans in doc_spans:
                segment_ids = []
                tokens = []
                if model_name == 'RoBERTa':
                    tokens.append('<s>')
                for q in ex['annotated_question']['word']:
                    segment_ids.append(0)
                    if model_name == 'RoBERTa' or model_name == 'SpanBERT':
                        tokens.append(q)
                        tokenizer.add_tokens([q])
                    else:
                        tokens.append(q.lower())
                        tokenizer.add_tokens([q.lower()])
                        # save_object([q.lower()], filename)

                if model_name == 'RoBERTa':
                    tokens.extend(['</s>', '</s>'])
                else:    
                    tokens.append('[SEP]')
                    segment_ids.append(0)
                
                tokenizer.add_tokens(paragraph[spans[0]:spans[0] + spans[1]])
                # save_object(paragraph[spans[0]:spans[0] + spans[1]], filename)
                tokens.extend(paragraph[spans[0]:spans[0] + spans[1]])
                segment_ids.extend([1] * spans[1])
                yes_index = len(tokens)
                tokens.append('yes')
                segment_ids.append(1)
                no_index = len(tokens)
                tokens.append('no')
                segment_ids.append(1)

                if spans[2] == 1:
                    tokens.append('<unknown>')
                    tokenizer.add_tokens(['<unknown>'])

                    # save_object(['<unknown>'], filename)

                    segment_ids.append(1)
                if model_name == 'RoBERTa':
                    tokens.append('</s>')
                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                converted_to_string = tokenizer.convert_ids_to_tokens(input_ids)
                input_ids.extend([0]*(512 - len(tokens)))
                input_mask.extend([0] * (512 - len(tokens)))
                segment_ids.extend([0] * (512 - len(tokens)))

                start = ex['answer_span'][0]
                end = ex['answer_span'][1]

                if start >= spans[0] and end <= spans[1]:
                    c_known+=1
                    start = question_length + 1 + start
                    end = question_length + 1 + end
                    
                else:
                    c_unknown+=1
                    start = len(tokens) - 1
                    end = len(tokens) - 1
                if ex['answer'] == 'yes' and tokens[start]!='yes':
                    start = yes_index
                    end = yes_index
                if ex['answer'] == 'no' and tokens[start]!='no':
                    start = no_index
                    end = no_index
                
                _example  = {'tokens': tokens, 'answer':tokens[start : end + 1],
                'actual_answer':ex['answer'] ,'input_tokens':input_ids, 'input_mask':input_mask,
                 'segment_ids':segment_ids, 'start':start, 'end':end,
                 'turn_id':ex['turn_id'],'paragraph_id':self.paragraphs[ex['paragraph_id']]['id']}
                self.chunked_examples.append(_example)
                #save_object(_example, sname)

        print("Chunk paragrapsh end.      tokenizer number: {} ".format(len(tokenizer))  ) 
        # if preprocess_step==PREPROCESS_STEP.SPLIT_DATA_AND_SAVE:
        self.save_tokenizer(tokenizer,data_set_range)
        self.save_coqa_dataset(data_set_range) #chunked_examples
     

    def save_tokenizer(self,tokenizer,data_set_range):
        if data_set_range==DATA_SET_RANGE.TRAIN_DATA:
            filename=TRAIN_TOKENIZER_VERSION1_FILENAME
            
        elif data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            filename=TRAIN_TOKENIZER_VERSION1_FILENAME
  
        else:
            filename=DEV_TOKENIZER_VERSION3_FILENAME
       
        torch.save(tokenizer,filename)
        print("save ",filename)
      
    def save_coqa_dataset(self, data_set_range):
        if data_set_range==DATA_SET_RANGE.TRAIN_DATA:
            filename=TRAIN_COQA_DATASET_VERSION1_FILENAME
        elif data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            filename=TRAIN_COQA_DATASET_VERSION1_FILENAME
           
        else:
            filename=DEV_COQA_DATASET_FILENAME
            
        torch.save(self,filename)
        print("save ",filename)


    def __len__(self):
        return len(self.chunked_examples)

    def __getitem__(self, idx):
        return self.chunked_examples[idx]


def load_tokenizer(parent_dir, preprocess_step,data_set_range,tokenizer):
    if preprocess_step==PROCESS_STEP.TRAIN__LOAD_ALL_DATA:   
        filename=DATA_PATH+parent_dir+"/dev_tokenizer_version3.pt"
    else:
        # if data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
        #     tokenizer=torch.load(TRAIN_TOKENIZER_VERSION1_FILENAME)
        # el
        if data_set_range==DATA_SET_RANGE.DEV_DATA:
            filename=DATA_PATH+parent_dir+"/train_tokenizer_version1.pt"
            
    tokenizer=torch.load(filename)
    print("load from ",filename)
    return tokenizer

class CustomDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.state = 0
        self.batch_state = 0
        self.current_view = []
        self.examples = [i for i in range(len(self.dataset))]
    
    def __len__(self):
        return math.ceil(len(self.examples)/self.batch_size)

    def prepare(self):
        shuffle(self.examples)
        self.state = 0
        self.batch_state = 0

    def restore(self, examples, state, batch_state):
        self.examples = examples
        self.state = state
        self.batch_state = batch_state
    
    def get(self):
        data_view = []
   
        for i in range(self.batch_size):
            if self.state + i < len(self.examples):
                example_index=self.examples[self.state + i]
                data_view.append(self.dataset[example_index])
                

        self.state += self.batch_size
        self.batch_state+=1
        return data_view 


if __name__=='__main__':
    from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('coqa.train.json')
