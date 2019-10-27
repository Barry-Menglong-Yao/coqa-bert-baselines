import json
import io
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from .timer import Timer
from collections import Counter, defaultdict
import numpy as np

def prepare_datasets(config, tokenizer_model):
    tokenizer = tokenizer_model[1].from_pretrained(tokenizer_model[2])
    trainset = CoQADataset(config['trainset'])
    trainset.chunk_paragraphs(tokenizer)
    trainloader = DataLoader(trainset, batch_size = config['batch_size'], shuffle = config['shuffle'], collate_fn=lambda x: x, pin_memory=True)
    devset = CoQADataset(config['devset'])
    devset.chunk_paragraphs(tokenizer)
    devloader = DataLoader(trainset, batch_size = config['batch_size'], shuffle = config['shuffle'], collate_fn=lambda x: x, pin_memory=True)
    return trainloader, devloader
def get_file_contents(filename, encoding='utf-8'):
    with io.open(filename, encoding=encoding) as f:
        content = f.read()
    f.close()
    return content


def read_json(filename, encoding='utf-8'):
    contents = get_file_contents(filename, encoding=encoding)
    return json.loads(contents)

class CoQADataset(Dataset):
    """CoQA dataset."""

    def __init__(self, filename):
        timer = Timer('Load %s' % filename)
        self.filename = filename
        paragraph_lens = []
        question_lens = []
        self.paragraphs = []
        self.examples = []
        self.vocab = Counter()
        dataset = read_json(filename)
        for paragraph in dataset['data']:
            history = []
            for qas in paragraph['qas']:
                qas['paragraph_id'] = len(self.paragraphs)
                temp = []
                n_history = len(history) #if config['n_history'] < 0 else min(config['n_history'], len(history))
                if n_history > 0:
                    for i, (q, a) in enumerate(history[-n_history:]):
                        d = n_history - i
                        temp.append('<Q{}>'.format(d))
                        temp.extend(q)
                        temp.append('<A{}>'.format(d))
                        temp.extend(a)
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
        timer.finish()
        self.chunked_examples = []
        

    def chunk_paragraphs(self, tokenizer):
        for i, ex in enumerate(self.examples):
            question_length = len(ex['annotated_question']['word'])
            if question_length > 350: # TODO provide from config
                continue
            doc_length_available = 512 - question_length - 1
            paragraph = self.paragraphs[ex['paragraph_id']]['annotated_context']['word']
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
                for q in ex['annotated_question']['word']:
                    segment_ids.append(0)
                    tokens.append(q)

                tokens.append('[SEP]')
                segment_ids.append(0)

                tokens.extend(paragraph[spans[0]:spans[0] + spans[1]])
                segment_ids.extend([1] * spans[1])
                if spans[2] == 1:
                    tokens.append('unknown')
                    segment_ids.append(1)
                input_mask = [1] * len(tokens)
                input_ids = tokenizer.convert_tokens_to_ids(tokens)
                input_ids.extend([0]*(512 - len(tokens)))
                input_mask.extend([0] * (512 - len(tokens)))
                segment_ids.extend([0] * (512 - len(tokens)))

                start = ex['answer_span'][0]
                end = ex['answer_span'][1]

                if start >= spans[0] and end <= spans[1]:
                    start = question_length + 1 + start
                    end = question_length + 1 + end
                else:
                    start = len(tokens) - 1
                    end = len(tokens) - 1
                _example  = {'tokens': tokens,'paragraph':paragraph, 'answer':tokens[start : end + 1], 'question':ex['annotated_question']['word'], 'span':ex['answer_span'] ,'input_tokens':input_ids, 'input_mask':input_mask, 'segment_ids':segment_ids, 'start':start, 'end':end}
                self.chunked_examples.append(_example)



    def __len__(self):
        return len(self.chunked_examples)

    def __getitem__(self, idx):
        return self.chunked_examples[idx]

if __name__=='__main__':
    from transformers import *
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = CoQADataset('/home/aniket/coqa-bert-baselines/data/coqa.train.json')
    dataset.chunk_paragraphs(tokenizer)
    dataloader = DataLoader(dataset, batch_size=64,shuffle=True, collate_fn=lambda x: x, pin_memory=True)
    for d in dataloader:
        print(d[0]['question'])
        print(d[0]['tokens'][d[0]['start']:d[0]['end']+1])
        print(d[0]['paragraph'][d[0]['span'][0]:d[0]['span'][1]+1])
        print(d[0]['input_mask'])
        break