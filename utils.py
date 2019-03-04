import json_lines
import os


def read_qa(data_dir, split, split_type):
    # lower cases text
    if split is None:
        files = [os.path.join(data_dir, '{}_{}_split{}.jsonl'.format(s, split_type, e))
                 for s, e in zip(['train', 'dev', 'test'], ['', '', '_no_answers'])]
    else:
        files = [os.path.join(data_dir, '{}_{}_split{}.jsonl'.format(split, split_type,
                                                                     '_no_answers' if split == 'test' else ''))]

    questions = []
    choices = []
    labels = []
    ans_dict = {'A': 0, 'B': 1, 'C': 2}
    for file in files:
        with open(file, 'rb') as f:  # opening file in binary(rb) mode
            for item in json_lines.reader(f):
                questions.append(item['question']['stem'].lower())

                ch = ['', '', '']
                for i in item['question']['choices']:
                    ch[ans_dict[i['label']]] = i['text'].lower()
                choices.append(ch)
                if 'answerKey' in item:
                    labels.append(item['answerKey'])
                else:
                    labels.append([])

    return questions, choices, labels
