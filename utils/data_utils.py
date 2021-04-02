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
TRAIN_COQA_DATASET_VERSION2_FILENAME="temp_data/train/train_coqa_dataset_version2_10000.pt"
#TODO "temp_data/dev/dev_coqa_dataset.pt"
DEV_COQA_DATASET_FILENAME=TRAIN_COQA_DATASET_VERSION2_FILENAME
TRAIN_TOKENIZER_VERSION1_FILENAME="temp_data/train/train_tokenizer_version1.pt"
TRAIN_TOKENIZER_VERSION2_FILENAME="temp_data/train/train_tokenizer_version2_10000.pt"
#TODO "temp_data/dev/dev_tokenizer_version3.pt"
DEV_TOKENIZER_VERSION3_FILENAME=TRAIN_TOKENIZER_VERSION2_FILENAME
 
number_of_part1=25000   #25000 for 4 hours. 125000 for all. 3 for test.

def prepare_datasets(config, tokenizer_model):
    print("Prepare dataset begin")
    tokenizer = tokenizer_model[1].from_pretrained(tokenizer_model[2])
    # if preprocess part 1
    data_set_range=DATA_SET_RANGE.TRAIN_DATA_SECOND_PART 
    # elif preprocess part 2
    # data_set_range=DATA_SET_RANGE.TRAIN_DATA_SECOND_PART

    preprocess_step=PREPROCESS_STEP.LOAD_ALL_DATA  
    trainloader=None 
    devloader=None 

    if preprocess_step==PREPROCESS_STEP.SPLIT_DATA_AND_SAVE:
        if data_set_range==DATA_SET_RANGE.DEV_DATA: 
            devset = CoQADataset(config['devset'])
            devset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,data_set_range)
            devloader = CustomDataLoader(devset, config['batch_size'])
        elif data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            trainset = torch.load(TRAIN_COQA_DATASET_VERSION1_FILENAME)
            trainset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,data_set_range)
            trainloader = CustomDataLoader(trainset, config['batch_size'])
        else:
            trainset = CoQADataset(config['trainset'])
            trainset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,data_set_range)
            trainloader = CustomDataLoader(trainset, config['batch_size'])
            
    else:
        trainset = torch.load(TRAIN_COQA_DATASET_VERSION2_FILENAME)
        # trainset = CoQADataset(config['trainset'])
        trainset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,DATA_SET_RANGE.TRAIN_DATA_SECOND_PART)
        trainloader = CustomDataLoader(trainset, config['batch_size'])
        devset = torch.load(DEV_COQA_DATASET_FILENAME)
        devset.chunk_paragraphs(tokenizer, config['model_name'],preprocess_step,DATA_SET_RANGE.DEV_DATA)
        devloader = CustomDataLoader(devset, config['batch_size'])  
 
    return trainloader, devloader, tokenizer

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
class PREPROCESS_STEP(Enum):
    SPLIT_DATA_AND_SAVE = 1  #is spliting data into multi parts and save them. will save from the last checkpoint
    LOAD_ALL_DATA = 2   #is in training step. Just load all dataset and cnt=0

class DATA_SET_RANGE(Enum):
    DEV_DATA=1
    TRAIN_DATA_FIRST_PART=2
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


    """
    1,self.paragraphs.append(paragraph)
    2,
    qas['annotated_question']['word'] is conversation history: Q1 A1 Q2 A2 ... current_Q
    self.examples.append(qas)
    3,self.vocab[w] += 1
    """
    def __init__(self, filename):
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
        '''
        number_before_break = 0
        for paragraph in tqdm(dataset['data']):
            if number_before_break == 5:
              break
            number_before_break += 1
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
        #timer.finish()



        #print(self.paragraphs)
        print("############################################")
        print(self.examples)
        self.chunked_examples = []
        def save_object(obj, filename):
            with open(filename, 'wb') as output:  # Overwrites any existing file.
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        #sname = 'temp_data/example.pkl'
        #save_object(self.examples, sname)
        #sname = 'temp_data/paragraph.pkl'

        #save_object(self.paragraphs, sname)
    #'''

    

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

    

    
     


    def chunk_paragraphs(self, tokenizer, model_name,preprocess_step,data_set_range):
        

        if preprocess_step==PREPROCESS_STEP.LOAD_ALL_DATA:
            tokenizer=self.load_tokenizer(preprocess_step,data_set_range,tokenizer)
        else:
            self.chunk_paragraphs_and_save(  tokenizer, model_name,preprocess_step,data_set_range)
        

         

    ##
    # generate input_tokens for BERT and save in self.chunked_examples  (can use minibatch)
    #  tokenizer.add_tokens()
    def chunk_paragraphs_and_save(self, tokenizer, model_name ,preprocess_step,data_set_range):
        #  when preprocess the second 50000 examples, we need firstly load the first 50000 token into tokenizer. Otherwise, the token_id of a word may change. For example, from 50010 to 10.
        tokenizer= self.load_tokenizer(preprocess_step,data_set_range,tokenizer)
        print("Chunk paragrapsh begin.      tokenizer number: {} ".format(len(tokenizer))  ) 
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

            if i >=total:
                break
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
                
                _example  = {'tokens': tokens, 'answer':tokens[start : end + 1],'actual_answer':ex['answer'] ,'input_tokens':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'start':start, 'end':end}
                self.chunked_examples.append(_example)
                #save_object(_example, sname)

        print("Chunk paragrapsh end.      tokenizer number: {} ".format(len(tokenizer))  ) 
        self.save_tokenizer(tokenizer,data_set_range)
        self.save_coqa_dataset(data_set_range) #chunked_examples
     

    def save_tokenizer(self,tokenizer,data_set_range):
        if data_set_range==DATA_SET_RANGE.TRAIN_DATA_FIRST_PART:
            torch.save(tokenizer,TRAIN_TOKENIZER_VERSION1_FILENAME)
        elif data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            torch.save(tokenizer,TRAIN_TOKENIZER_VERSION2_FILENAME)
        else:
            torch.save(tokenizer,DEV_TOKENIZER_VERSION3_FILENAME)
      
    def save_coqa_dataset(self, data_set_range):
        if data_set_range==DATA_SET_RANGE.TRAIN_DATA_FIRST_PART:
            torch.save(self,TRAIN_COQA_DATASET_VERSION1_FILENAME)
        elif data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
            torch.save(self,TRAIN_COQA_DATASET_VERSION2_FILENAME)
        else:
            torch.save(self,DEV_COQA_DATASET_FILENAME)

    def load_tokenizer(self,preprocess_step,data_set_range,tokenizer):
        if preprocess_step==PREPROCESS_STEP.LOAD_ALL_DATA:
            if data_set_range==DATA_SET_RANGE.DEV_DATA: 
                tokenizer=torch.load(DEV_TOKENIZER_VERSION3_FILENAME)
            else:
                tokenizer=torch.load(TRAIN_TOKENIZER_VERSION2_FILENAME)
        else:
            if data_set_range==DATA_SET_RANGE.TRAIN_DATA_SECOND_PART:
                tokenizer=torch.load(TRAIN_TOKENIZER_VERSION1_FILENAME)
            elif data_set_range==DATA_SET_RANGE.DEV_DATA:
                tokenizer=torch.load(TRAIN_TOKENIZER_VERSION2_FILENAME)
        
        
        return tokenizer

    def __len__(self):
        return len(self.chunked_examples)

    def __getitem__(self, idx):
        return self.chunked_examples[idx]


class CustomDataLoader:
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.state = 0
        self.batch_state = 0
        self.examples = [i for i in range(len(self.dataset))]
        self.current_view = []
    
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
                data_view.append(self.dataset[self.examples[self.state + i]])
        self.state += self.batch_size
        self.batch_state+=1
        return data_view


if __name__=='__main__':
    from transformers import *
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('coqa.train.json')
