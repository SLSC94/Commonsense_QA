import numpy as np
import torch
import re
import requests
import os

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import read_qa


class ConceptNet:
    questions = chocies = labels = vocab = None

    def __init__(self, bert_model='bert-base-uncased', config_file='bert/bert_config.json', qa_model=None):
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)

        # for embeddings
        if qa_model is None:
            model = BertForMultipleChoice.from_pretrained(bert_model,
                                                          cache_dir=os.path.join(PYTORCH_PRETRAINED_BERT_CACHE,
                                                                                 'distributed_{}'.format(-1)),
                                                          num_choices=3)
        else:
            config = BertConfig(config_file)
            model = BertForMultipleChoice(config, num_choices=3)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model.load_state_dict(torch.load(qa_model, map_location=dev))

        self.model = model

    def load_file(self, split=None, split_type='rand'):
        if split is None:
            splits = ['train', 'dev', 'test']
        else:
            splits = [split]

        questions = []
        choices = []
        labels = []
        for s in splits:
            q, c, a = read_qa('data', s, split_type)
            questions += q
            choices += c
            labels += a

        self.questions = questions
        self.choices = choices
        self.labels = labels

    def build_vocab(self):
        assert self.questions is not None, 'Load questions from data file first. '
        regex = r'\b\w+\b'
        vocab = re.findall(r'\b\w+\b', ' '.join(self.questions + list(np.reshape(self.choices, -1))))
        vocab = list(set(vocab))
        self.vocab = vocab

    def get_triplet(self, word, limit=20):
        # get all of one word's triplet
        lim = int(limit)  # max number of relations returned for each word

        # This is the API.
        obj = requests.get('http://api.conceptnet.io/c/en/' + word + '?offset=0&limit={}'.format(lim)).json()
        relations = []
        term = '/c/en/' + word
        if 'error' not in obj.keys():  # eliminate words not found in ConceptNet
            for j in obj['edges']:
                if 'language' not in j['start'] or 'language' not in j['end']:
                    continue
                if j['start']['language'] == 'en' and j['end']['language'] == 'en':
                    rel = [j['start']['label'],
                           j['weight'],
                           j['end']['label'],
                           j['rel']['label']]

                    if j['start']['term'] == term:
                        rel[0] = ''
                    else:
                        rel[2] = ''

                    relations.append(rel)
            return [word, relations]

    def get_triplets_multithread(self, vocab=None, n_jobs=16):
        # using multithreading
        # returns dict for quick access to neighbors of word

        if vocab is None:
            vocab = self.vocab
        assert vocab is not None, 'Pass vocab of words in.'
        assert len(vocab) <= 6000, 'Limit to 6000 tries.'

        from joblib import Parallel, delayed
        output = Parallel(n_jobs=n_jobs)(delayed(self.get_triplet)(i) for i in vocab)
        output = [x for x in output if x is not None]
        return dict(output)

    def get_source_concept(self, relations):
        # TODO: loop thru each question and choice, find word in question that has all 3 choices as neighbours
        pass

    def similarity(self, w1, w2):
        # use embedding from before finetuning
        assert self.tokenizer is not None, 'Unknown tokenizer'
        ids = self.tokenizer.convert_tokens_to_ids([w1, w2])
        emb = self.model.bert.embeddings(torch.tensor([ids]))[0]
        return int(emb[0] @ emb[1])

    def construct_subgraph(self, qn, choices, k=10, max_n=50):
        '''

        :param qn:
        :param choices:
        :param k: beam search width
        :param max_n: number of nodes in graph
        :return: adj matrix of size max_n
        '''
        assert self.questions is not None, 'Load data file first!'
        # TODO
        '''
            1) identify source concept S and relevant concepts R in question (using attention or n-gram)
            2) for each layer, rank each connected node's similarity to R and keep top k
            3) need to handle cycles in graph
            4) stop at max_n nodes
            5) return weighted adj matrix
        '''
        return np.eye(max_n)
