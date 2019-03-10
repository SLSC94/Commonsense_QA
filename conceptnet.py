import numpy as np
import torch
import re
import requests
import os
import json_lines
import json

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForMultipleChoice, BertConfig
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from utils import read_qa


class ConceptNet:
    questions = choices = labels = vocab = None

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

        model.eval();
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
        vocab = re.findall(r'\b\w+\b', ' '.join(self.questions + list(np.reshape(self.choices, -1))))
        vocab = list(set(vocab))
        self.vocab = vocab

    def get_triplet(self, word, limit=20):
        # get all of one word's triplet
        lim = int(limit)  # max number of relations returned for each word

        # This is the API.
        obj = requests.get('http://api.conceptnet.io/c/en/' + word + '?offset=0&limit={}'.format(lim)).json()
        relations = []

        if 'error' not in obj.keys():  # eliminate words not found in ConceptNet
            for j in obj['edges']:
                if 'language' not in j['start'] or 'language' not in j['end']:
                    continue
                if j['start']['language'] == 'en' and j['end']['language'] == 'en':
                    rel = [j['start']['label'],
                           j['weight'],
                           j['end']['label'],
                           j['rel']['label']]

                    relations.append(rel)
            return [word, relations]
    
    def decoder(self, dictionary):
        for key in dictionary.keys():
            items = dictionary[key]
            dictionary[key] = [tuple(item) for item in items]
        return dictionary
    
    def break_sentence(self, sentence): 
        '''
        This breaks a sentence into a list of individual words
        e.g.
        'I like dogs' -> ['i', 'like', 'dogs']
        '''
        regex = r'\b\w+\b'
        return re.findall(regex, sentence.lower())

    def load_triplets(self, files=['data/train_triplets.txt',
                                   'data/dev_triplets.txt',
                                   'data/test_triplets.txt']):
        '''
        load in the triplets directly - separated just in case. Joining them is easy.
        '''
        triplets = {}
        for i in files:
            triplets.update(self.decoder(json.load(open(i))))

        self.triplets = triplets

    def get_triplets_multithread(self, vocab=None, n_jobs=16):
        # using multithreading
        # returns dict for quick access to neighbors of word

        if vocab is None:
            vocab = self.vocab[0:10]
        assert vocab is not None, 'Pass vocab of words in.'
        assert len(vocab) <= 6000, 'Limit to 6000 tries.'

        from joblib import Parallel, delayed
        output = Parallel(n_jobs=n_jobs)(delayed(self.get_triplet)(i) for i in vocab)
        output = [x for x in output if x is not None]
        return dict(output)

    def get_source_concept(self):
        # TODO: loop thru each question and choice, find word in question that has all 3 choices as neighbours
        '''
        we will put the question and choices in a list form:
        e.g. 
        Q: "what is a happy day", Choices: "birthday", "funeral", "exam day"
        becomes
        ['what', 'is', 'a', 'happy', 'day'], [['birthday'], ['funeral'], ['exam', 'day']]
        
        We then look at each word in the list ['what', 'is', 'a', 'happy', 'day'] and
        come up with their neighbours, i.e.
        list1 = [N('what'), N('is'), N('a'), N('happy'), N('day')]
        
        Do the same for the second list to get 
        list2 = [N('birthday'), N('funeral'), {N('exam'), N('day')} ]
        
        and for each set in list1, count how many times there is an intersection with 
        each of the three sets in list2. 
        '''

        assert self.questions is not None, 'load data first'

        #we need to get rid of common words:
        from nltk.corpus import stopwords
        SW = set(stopwords.words('english'))
        
        
        
        self.source_concept = []
        for i in range(len(self.questions)):
            Q = self.questions[i]
            C = self.choices[i]
            
            #this counts how many times each word in the question has a neighbor which is in
            #the choices. min is 0, max is 3
            counter = np.zeros(len(self.break_sentence(Q)))
            flag = 0
            for j,word in enumerate(self.break_sentence(Q) ):
                #all the triplets associated with this word
                
                #only if word is not a stop words
                if word not in SW:
                #in case the words are not in the vocabulary
                    try:
                        node_set = self.get_neighbours(word)
                        for k in range(3):
                            choice_set = set()
                            for wd in self.break_sentence(C[k]) :
                                try:
                                    choice_set = choice_set.union(self.get_neighbours(wd))
                                except:
                                    if flag == 0:
                                        print("Iteration: ", i, " ,Danger Word: ", wd)
                                        flag = 1

                            if len(node_set.intersection(choice_set) ) != 0:
                                counter[j] += 1

                    except:
                        "Error!"
                        counter[j] = 0
                else:
                    counter[j] = -1 #so that we do not choose any stop words. 
            self.source_concept.append(self.break_sentence(Q)[np.argmax(counter)] )


        
        
    def get_neighbours(self, word):
        triplets = self.triplets[word]
        start_nodes = set([triplet[0] for triplet in triplets])
        end_nodes = set([triplet[2] for triplet in triplets])
        node_set = start_nodes.union(end_nodes)
        return node_set
    
    
    def similarity(self, w1, w2):
        # use embedding from before finetuning
        assert self.tokenizer is not None, 'Unknown tokenizer'
        ids = self.tokenizer.convert_tokens_to_ids([w1, w2])
        emb = self.model.bert.embeddings.word_embeddings(torch.tensor([ids]))[0]
        return float(emb[0] @ emb[1])

    def construct_subgraph(self, k=10, max_n=50):
        '''
        Will create both a list of adjacency matrices and a list of feature vectors
        '''
        
        
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
        HOW THE SUBGRAPH WILL BE CONSTRUCTED
            1) Source concept S, and choices A, B, C will be the only nodes in the graph for now
            2) The weight between each pair of nodes will be the dot product between their
               average word embedding (since choices can have multiple words)
            3) return weighted adj matrix
        '''
        
        '''
        This will also contain the features to pass into the CGN
        '''
        
        self.problem_words = 0
        self.adjacency_mat = []
        self.Gfeature_mat = []
        for i in range(len(self.questions)):
            S = self.source_concept[i]
            C = self.choices[i]
            
            emb = self.get_avg_embedding(S)
            for j in range(3):
                C_emb = self.get_avg_embedding(C[j])
                emb = torch.cat((emb, C_emb), dim = 0)
            
            adj_mat = torch.matmul(emb, torch.t(emb))
            self.adjacency_mat.append(adj_mat)
            self.Gfeature_mat.append(emb)

    def get_avg_embedding(self, words, verbose=True):
        list_words = self.break_sentence(words)
        try:
            ids_words = self.tokenizer.convert_tokens_to_ids(list_words)
            emb_words = self.model.bert.embeddings.word_embeddings.forward(torch.tensor(ids_words))
        except:
            emb_words = torch.ones([1,768])
            if verbose:
                self.problem_words += 1
                print(list_words)
        avg_emb = emb_words.mean(dim=0)[None,:]
        '''normalise'''
        avg_emb = avg_emb/avg_emb.norm()
        return avg_emb