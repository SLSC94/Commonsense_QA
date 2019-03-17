import torch.nn as nn
from sklearn.metrics import accuracy_score
from gpt.model_pytorch import TransformerModel, DoubleHeadModel, load_openai_pretrained_model, DEFAULT_CONFIG
from gpt.loss import MultipleChoiceLossCompute
from gpt.text_utils import TextEncoder
from gpt.utils import iter_data, ResultLogger
from gpt.opt import OpenAIAdam
import jsonlines
import torch, datetime as dt
import numpy as np
from sklearn.utils import shuffle

import os
import argparse


def init_parser(PATH):
    parser = argparse.ArgumentParser()
    parser.add_argument('--desc', type=str, help="Description")
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--log_dir', type=str, default=os.path.join(PATH, 'log/'))
    parser.add_argument('--save_dir', type=str, default=os.path.join(PATH, 'save/'))
    parser.add_argument('--data_dir', type=str, default=os.path.join(PATH, 'data/'))
    parser.add_argument('--submission_dir', type=str, default=os.path.join(PATH, 'submission/'))
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--analysis', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_iter', type=int, default=3)
    parser.add_argument('--n_batch', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--lr', type=float, default=6.25e-5)
    parser.add_argument('--lr_warmup', type=float, default=0.002)
    parser.add_argument('--n_ctx', type=int, default=512)
    parser.add_argument('--n_embd', type=int, default=768)
    parser.add_argument('--n_head', type=int, default=12)
    parser.add_argument('--n_layer', type=int, default=12)
    parser.add_argument('--embd_pdrop', type=float, default=0.1)
    parser.add_argument('--attn_pdrop', type=float, default=0.1)
    parser.add_argument('--resid_pdrop', type=float, default=0.1)
    parser.add_argument('--clf_pdrop', type=float, default=0.1)
    parser.add_argument('--l2', type=float, default=0.01)
    parser.add_argument('--vector_l2', action='store_true')
    parser.add_argument('--opt', type=str, default='adam')
    parser.add_argument('--afn', type=str, default='gelu')
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--encoder_path', type=str, default=os.path.join(PATH, 'model/encoder_bpe_40000.json'))
    parser.add_argument('--bpe_path', type=str, default=os.path.join(PATH, 'model/vocab_40000.bpe'))
    parser.add_argument('--n_transfer', type=int, default=12)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--b1', type=float, default=0.9)
    parser.add_argument('--b2', type=float, default=0.999)
    parser.add_argument('--e', type=float, default=1e-8)
    parser.add_argument('--n_valid', type=int, default=374)

    return parser


def read_data(f):
  x = []
  choices = []
  y = []
  ans_dict = {'A':0,'B':1,'C':2}
  with jsonlines.open(f) as reader:
    for obj in reader:
      x.append(obj['question']['stem'])
      if 'answerKey' in obj:
        y.append(ans_dict[obj['answerKey']])

      choices+=[['','','']]
      for i in obj['question']['choices']:
        choices[-1][ans_dict[i['label']]] = i['text']

  choices = np.array(choices)
  out = [x]+[choices[:,i] for i in range(choices.shape[1])]
  if len(y)>0: out+=[y]
  return out


def encode_dataset(*splits, encoder):
  encoded_splits = []
  for split in splits:
    fields = []
    for field in split:
      if isinstance(field[0], str):
        field = encoder.encode(field)
      fields.append(field)

    encoded_splits.append(fields)
  return encoded_splits


def finetuning(PATH, args, files, output_dir):
    n_ctx = args.n_ctx  # max sentence length

    encoder_path = os.path.join(PATH, 'gpt/model/encoder_bpe_40000.json')
    bpe_path = os.path.join(PATH, 'gpt/model/vocab_40000.bpe')

    text_encoder = TextEncoder(encoder_path, bpe_path)
    encoder = text_encoder.encoder
    n_vocab = len(text_encoder.encoder)

    # assign new tokens
    encoder['_start_'] = len(encoder)
    encoder['_delimiter_'] = len(encoder)
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']

    n_special = 3
    max_len = n_ctx // 2 - 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print("device", device, "n_gpu", n_gpu)

    data_sets = [read_data(f) for f in files]

    ((trQ, trX1, trX2, trX3, trY),
     (vaQ, vaX1, vaX2, vaX3, vaY),
     (teQ, teX1, teX2, teX3)) = encode_dataset(*data_sets, encoder=text_encoder)

    n_ctx = min(max(
        [len(q[:max_len]) + max(len(x1[:max_len]),
                                len(x2[:max_len]),
                                len(x3[:max_len])) for q, x1, x2, x3 in zip(trQ, trX1, trX2, trX3)]
        + [len(q[:max_len]) + max(len(x1[:max_len]),
                                  len(x2[:max_len]),
                                  len(x3[:max_len])) for q, x1, x2, x3 in zip(vaQ, vaX1, vaX2, vaX3)]
        + [len(q[:max_len]) + max(len(x1[:max_len]),
                                  len(x2[:max_len]),
                                  len(x3[:max_len])) for q, x1, x2, x3 in zip(teQ, teX1, teX2, teX3)]
    ) + 3, n_ctx)

    vocab = n_vocab + n_special + n_ctx

    def transform_qa(Q, X1, X2, X3):
        n_batch = len(Q)
        xmb = np.zeros((n_batch, 3, n_ctx, 2), dtype=np.int32)
        mmb = np.zeros((n_batch, 3, n_ctx), dtype=np.float32)
        start = encoder['_start_']
        delimiter = encoder['_delimiter_']
        for i, (q, x1, x2, x3), in enumerate(zip(Q, X1, X2, X3)):
            x11 = [start] + q[:max_len] + [delimiter] + x1[:max_len] + [clf_token]
            x12 = [start] + q[:max_len] + [delimiter] + x2[:max_len] + [clf_token]
            x13 = [start] + q[:max_len] + [delimiter] + x3[:max_len] + [clf_token]
            l11 = len(x11)
            l12 = len(x12)
            l13 = len(x13)
            xmb[i, 0, :l11, 0] = x11
            xmb[i, 1, :l12, 0] = x12
            xmb[i, 2, :l13, 0] = x13
            mmb[i, 0, :l11] = 1
            mmb[i, 1, :l12] = 1
            mmb[i, 2, :l13] = 1
        # Position information that is added to the input embeddings in the TransformerModel
        xmb[:, :, :, 1] = np.arange(n_vocab + n_special, n_vocab + n_special + n_ctx)
        return xmb, mmb

    trX, trM = transform_qa(trQ, trX1, trX2, trX3)
    vaX, vaM = transform_qa(vaQ, vaX1, vaX2, vaX3)

    n_train = len(trY)
    n_valid = len(vaY)
    n_batch_train = 8
    n_updates_total = (n_train // n_batch_train) * args.n_iter

    dh_model = DoubleHeadModel(args, clf_token, 'multiple_choice', vocab, n_ctx)

    criterion = nn.CrossEntropyLoss(reduce=False)
    model_opt = OpenAIAdam(dh_model.parameters(),
                           lr=args.lr,
                           schedule=args.lr_schedule,
                           warmup=args.lr_warmup,
                           t_total=n_updates_total,
                           b1=args.b1,
                           b2=args.b2,
                           e=args.e,
                           l2=args.l2,
                           vector_l2=args.vector_l2,
                           max_grad_norm=args.max_grad_norm)

    compute_loss_fct = MultipleChoiceLossCompute(criterion,
                                                 criterion,
                                                 args.lm_coef,
                                                 model_opt)

    load_openai_pretrained_model(dh_model.transformer, n_ctx=n_ctx, n_special=n_special,
                                 path=os.path.join(PATH, 'gpt/model/'), path_names=os.path.join(PATH, 'gpt/'))

    dh_model.to(device)
    dh_model = nn.DataParallel(dh_model)

    n_updates = 0
    n_epochs = 0

    log_dir = os.path.join(PATH, args.log_dir)
    desc = dt.datetime.today().strftime('%Y%m%d')

    logger = ResultLogger(path=os.path.join(log_dir, '{}.jsonl'.format(desc)), **args.__dict__)

    save_dir = args.save_dir

    def run_epoch():
        for xmb, mmb, ymb in iter_data(*shuffle(trX, trM, trY, random_state=np.random),
                                       n_batch=n_batch_train, truncate=True, verbose=True):
            global n_updates
            dh_model.train()
            XMB = torch.tensor(xmb, dtype=torch.long).to(device)
            YMB = torch.tensor(ymb, dtype=torch.long).to(device)
            MMB = torch.tensor(mmb).to(device)
            lm_logits, clf_logits = dh_model(XMB)
            compute_loss_fct(XMB, YMB, MMB, clf_logits, lm_logits)
            n_updates += 1
            if n_updates % 100 == 0:
                log(save_dir, desc)

    def log(save_dir, desc):
        global best_score
        print("Logging")
        tr_logits, tr_cost = iter_apply(trX[:n_valid], trM[:n_valid], trY[:n_valid])
        va_logits, va_cost = iter_apply(vaX, vaM, vaY)
        tr_cost = tr_cost / len(trY[:n_valid])
        va_cost = va_cost / n_valid
        tr_acc = accuracy_score(trY[:n_valid], np.argmax(tr_logits, 1)) * 100.
        va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
        logger.log(n_epochs=n_epochs, n_updates=n_updates, tr_cost=tr_cost, va_cost=va_cost, tr_acc=tr_acc,
                   va_acc=va_acc)
        print('%d %d %.3f %.3f %.2f %.2f' % (n_epochs, n_updates, tr_cost, va_cost, tr_acc, va_acc))

    def iter_apply(Xs, Ms, Ys):
        logits = []
        cost = 0
        with torch.no_grad():
            dh_model.eval()
            for xmb, mmb, ymb in iter_data(Xs, Ms, Ys, n_batch=n_batch_train, truncate=False, verbose=True):
                n = len(xmb)
                XMB = torch.tensor(xmb, dtype=torch.long).to(device)
                YMB = torch.tensor(ymb, dtype=torch.long).to(device)
                MMB = torch.tensor(mmb).to(device)
                _, clf_logits = dh_model(XMB)
                clf_logits *= n
                clf_losses = compute_loss_fct(XMB, YMB, MMB, clf_logits, only_return_losses=True)
                clf_losses *= n
                logits.append(clf_logits.to("cpu").numpy())
                cost += clf_losses.sum().item()
            logits = np.concatenate(logits, 0)
        return logits, cost

    best_score = 0
    for i in range(args.n_iter):
        print("running epoch", i)
        run_epoch()
        n_epochs += 1

    va_logits, va_cost = iter_apply(vaX, vaM, vaY)
    va_acc = accuracy_score(vaY, np.argmax(va_logits, 1)) * 100.
    tr_logits, tr_cost = iter_apply(trX, trM, trY)
    tr_acc = accuracy_score(trY, np.argmax(tr_logits, 1)) * 100.

    print(tr_acc, tr_cost, va_acc, va_cost)

    # checkpoint
    torch.save({
        'epoch': n_epochs,
        'model_state_dict': dh_model.state_dict(),
        'optimizer_state_dict': model_opt.state_dict(),
        'loss': tr_cost,
        'va_acc': va_acc
    }, os.path.join(output_dir, 'gpt-{}.pt'.format(dt.datetime.now().strftime('%Y%m%d%H%M%S'))))
