"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import copy
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import chess
import torch
import torch.nn as nn
from torch.nn import functional as F
from ast import literal_eval
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from collections import defaultdict


CUDA_LAUNCH_BLOCKING = 1

# -----------------------------------------------------------------------------


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # define attention scores attribute to record scores for later
        self.att_scores = None

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        ### collect attention scores
        self.att_scores = copy.copy(att)
        att = self.attn_dropout(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.att_scores = None
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        # collect attention scores
        self.att_scores = self.attn.att_scores
        return x

class GPT(nn.Module):
    """ GPT Language Model """

    @staticmethod
    def get_default_config():
        C = CfgNode()
        # either model_type or (n_layer, n_head, n_embd) must be given in the config
        C.device = 'auto'
        C.model_type = 'chess-gpt'
        C.n_layer = None
        C.n_head = None
        C.n_embd =  None
        # these options must be filled in externally
        C.vocab_size = None
        C.block_size = None
        # dropout hyperparameters
        C.embd_pdrop = 0.1
        C.resid_pdrop = 0.1
        C.attn_pdrop = 0.1
        return C

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.block_size = config.block_size

        type_given = config.model_type is not None
        params_given = all([config.n_layer is not None, config.n_head is not None, config.n_embd is not None])
        assert type_given ^ params_given # exactly one of these (XOR)
        if type_given:
            # translate from model_type to detailed configuration
            config.merge_from_dict({
                'ChessGPTNano':   dict(n_layer=10, n_head=8, n_embd=32),     # 100k
                'ChessGPTSmall':  dict(n_layer=6, n_head=8, n_embd=64),     # 300k params
                'ChessGPT':       dict(n_layer=12, n_head=8, n_embd=64),    # 600k params
                'ChessGPT2':      dict(n_layer=4, n_head=8, n_embd=128),    # 800k params
                'ChessGPTMed':    dict(n_layer=10, n_head=16, n_embd=128),  # 2M params
                'ChessGPTLarge':  dict(n_layer=12, n_head=8, n_embd=128),   # 2.4M params
                'ChessGPTLarger': dict(n_layer=6, n_head=8, n_embd=256)
            }[config.model_type])

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.embd_pdrop),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params/1e6,))

        # define attention scores list (each entry is attention scores in a block)
        self.att_scores = []
        self.config = config

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx, targets=None):
        self.att_scores = []
        device = idx.device
        b, t = idx.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        # forward the GPT model itself
        for block in self.transformer.h:
            x = block(x)
            # collect attention scores for block
            self.att_scores.append(block.att_scores)
        x = self.transformer.ln_f(x)
        #logits = self.lm_head(x)

        # if we are given some desired targets also calculate the loss
        #loss = None
        #if targets is not None:
        #    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        #return logits, loss
        return x
# utility class
class CfgNode:
    """ a lightweight configuration class inspired by yacs """

    # TODO: convert to subclass from a dict like in yacs?
    # TODO: implement freezing to prevent shooting of own foot
    # TODO: additional existence/override checks when reading/writing params?

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return self._str_helper(0)

    def _str_helper(self, indent):
        """ need to have a helper to support nested indentation for pretty printing """
        parts = []
        for k, v in self.__dict__.items():
            if isinstance(v, CfgNode):
                parts.append("%s:\n" % k)
                parts.append(v._str_helper(indent + 1))
            else:
                parts.append("%s: %s\n" % (k, v))
        parts = [' ' * (indent * 4) + p for p in parts]
        return "".join(parts)

    def to_dict(self):
        """ return a dict representation of the config """
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.__dict__.items()}

    def merge_from_dict(self, d):
        self.__dict__.update(d)

    def merge_from_args(self, args):
        """
        update the configuration from a list of strings that is expected
        to come from the command line, i.e. sys.argv[1:].
        The arguments are expected to be in the form of `--arg=value`, and
        the arg can use . to denote nested sub-attributes. Example:
        --model.n_layer=10 --trainer.batch_size=32
        """
        for arg in args:

            keyval = arg.split('=')
            assert len(keyval) == 2, "expecting each override arg to be of form --arg=value, got %s" % arg
            key, val = keyval  # unpack

            # first translate val into a python object
            try:
                val = literal_eval(val)
                """
                need some explanation here.
                - if val is simply a string, literal_eval will throw a ValueError
                - if val represents a thing (like an 3, 3.14, [1,2,3], False, None, etc.) it will get created
                """
            except ValueError:
                pass

            # find the appropriate object to insert the attribute into
            assert key[:2] == '--'
            key = key[2:]  # strip the '--'
            keys = key.split('.')
            obj = self
            for k in keys[:-1]:
                obj = getattr(obj, k)
            leaf_key = keys[-1]

            # ensure that this attribute exists
            assert hasattr(obj, leaf_key), f"{key} is not an attribute that exists in the config"

            # overwrite the attribute
            print("command line overwriting config attribute %s with %s" % (key, val))
            setattr(obj, leaf_key, val)


class WinTrainer:
    @staticmethod
    def get_default_config(batch=256, lr=3e-4):
        C = CfgNode()
        # device to train on
        C.device = 'auto'
        # dataloder parameters
        C.num_workers = 0
        # optimizer parameters
        C.max_iters = None
        C.batch_size = batch
        C.learning_rate = lr
        C.betas = (0.9, 0.95)  # original (0.9, 0.95)
        C.weight_decay = 1e-5  # 0.1 only applied on matmul weights
        C.grad_norm_clip = 1.0
        return C

    def __init__(self, config, model, trainset, testset=None, type='binaryclass'):
        self.config = config
        self.model = model
        self.optimizer = None
        self.trainset = trainset
        self.testset = testset
        self.callbacks = defaultdict(list)
        self.loss_table = {}
        self.loss_table["time"] = []
        self.loss_table["loss"] = []
        self.type = type

        # determine the device we'll train on
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device
        self.model = self.model.to(self.device)
        print("running on device", self.device)

        # variables that will be assigned to trainer class later for logging etc
        self.iter_num = 0
        self.iter_time = 0.0
        self.iter_dt = 0.0

    def add_callback(self, onevent: str, callback):
        self.callbacks[onevent].append(callback)

    def set_callback(self, onevent: str, callback):
        self.callbacks[onevent] = [callback]

    def trigger_callbacks(self, onevent: str):
        for callback in self.callbacks.get(onevent, []):
            callback(self)

    def run(self, classifier=None, backbone=None, final="last_token"):
        print("training")
        model, config = self.model, self.config
        model.train()

        # configure output head for model

        if self.type == "binaryclass":
            if classifier is None:
                classifier = nn.Sequential(nn.Linear(model.config.n_embd, model.config.n_embd),
                                         nn.BatchNorm1d(model.config.n_embd),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(model.config.n_embd, model.config.n_embd),
                                         nn.BatchNorm1d(model.config.n_embd),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(model.config.n_embd, model.config.n_embd),
                                         nn.BatchNorm1d(model.config.n_embd),
                                         nn.LeakyReLU(inplace=True),
                                         nn.Linear(model.config.n_embd, 1)).to(self.device)
            classifier.train().to(self.device)
            loss_fn = nn.BCEWithLogitsLoss()
        

        # setup the optimizer
        self.optimizer = model.configure_optimizers(config)
        if classifier is not None:
            self.clfoptim = torch.optim.AdamW(classifier.parameters(), lr=self.config.learning_rate  # )
                                              , betas=self.config.betas, weight_decay=self.config.weight_decay)

        # setup the dataloader
        train_loader = DataLoader(
            self.trainset,
            sampler=torch.utils.data.RandomSampler(self.trainset, replacement=True),  # num_samples=int(1e10)),
            shuffle=False,
            # pin_memory=True,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        if self.testset is not None:
            test_loader = DataLoader(self.testset, shuffle=False, batch_size=config.batch_size,
                                     num_workers=config.num_workers)

        self.iter_num = 1
        data_iter = iter(train_loader)
        while True:
            #print("iter: ", self.iter_num)
            # fetch the next batch (x, y) and re-init iterator if needed
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)
            batch = [t.to(self.device) for t in batch]
            x, y = batch

            # forward the model
            if backbone is not None:
                backbone.eval()
                with torch.no_grad():
                    x = backbone(x)
            if final == "last_token":
                encoding = model(x)[:, -1, :]  # version that takes last token output
            elif final == "mean_tokens":
                encoding = torch.mean(model(x), dim=1)  # version that takes mean of outputs
            elif final == "concat_tokens":
                encoding = torch.flatten(model(x), start_dim=1, end_dim=2)  # version that concatenates outputs

            # prediction
            if self.type == "binaryclass":
                #predict = torch.sigmoid(classifier(encoding)).squeeze()
                predict = classifier(encoding).squeeze()
                self.loss = loss_fn(predict, y.float())
            
            ### save losses
            if self.iter_num % 100 == 0:
                self.loss_table["loss"].append(self.loss)
                self.loss_table["time"].append(self.iter_num)
                print(f"iter: {self.iter_num},  loss: {self.loss}")
                #for name, param in model.named_parameters():
                #    print(name, param.grad.abs().sum())

            # backprop and update the parameters
            self.optimizer.zero_grad(set_to_none=True)
            if classifier is not None:
                self.clfoptim.zero_grad(set_to_none=True)
            self.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_norm_clip)
            # torch.nn.utils.clip_grad_norm_(classifier.parameters(), config.grad_norm_clip)
            if classifier is not None:
                self.clfoptim.step()
            self.optimizer.step()

            self.trigger_callbacks('on_batch_end')
            self.iter_num += 1

            # test accuracy during training
            if self.iter_num % 2000 == 0 and self.testset is not None:
                print("intermediate evaluation")
                model.eval()
                classifier.eval()
                accuracy = 0
                with torch.no_grad():
                    for x, y in test_loader:
                        if backbone is not None:
                            backbone.eval()
                            with torch.no_grad():
                                x = backbone(x)
                        if final == "mean_tokens":
                            encoding = torch.mean(model(x), dim=1)
                        elif final == "concat_tokens":
                            encoding = torch.flatten(model(x), start_dim=1, end_dim=2)
                        elif final == "last_token":
                            encoding = model(x)[:, -1, :]
                        # make prediction using classifier
                        if self.type == "binaryclass":
                            proba = torch.sigmoid(classifier(encoding))
                            predict = proba.round().squeeze().to(self.device)
                            if torch.all(predict == 0):
                                print("all zero prediction")
                            elif torch.all(predict == 1):
                                print("all positive prediction")
                            y = y.round().squeeze().to(self.device)
                            accuracy += torch.eq(predict, y).sum().item()

                accuracy /= len(self.testset)
                print(f"accuracy: ", accuracy)
                model.train()
                classifier.train()

            # save checkpoint
            if self.iter_num % 100 == 0:
                state_dict = model.state_dict()
                torch.save(state_dict, f"drive/MyDrive/EncoderCheckpoint.pt")
                state_dict = classifier.state_dict()
                torch.save(state_dict, f"drive/MyDrive/ClfCheckpoint.pt")
                
            # termination conditions
            if config.max_iters is not None and self.iter_num >= config.max_iters:
                break

        return model, classifier, self.loss_table


def board_encoder(position):
    board = chess.Board(position)
    bits = []
    for i in range(64):
        p = board.piece_at(i)
        if p is None:
            bits.append(0)
        elif board.turn == chess.WHITE:
            if p.symbol() == "P":
                bits.append(1)
            elif p.symbol() == "R":
                bits.append(2)
            elif p.symbol() == "N":
                bits.append(3)
            elif p.symbol() == "B":
                bits.append(4)
            elif p.symbol() == "Q":
                bits.append(5)
            elif p.symbol() == "K":
                bits.append(6)
            elif p.symbol() == "p":
                bits.append(7)
            elif p.symbol() == "r":
                bits.append(8)
            elif p.symbol() == "n":
                bits.append(9)
            elif p.symbol() == "b":
                bits.append(10)
            elif p.symbol() == "q":
                bits.append(11)
            elif p.symbol() == "k":
                bits.append(12)
        elif board.turn == chess.BLACK:
            if p.symbol() == "P":
                bits.append(7)
            elif p.symbol() == "R":
                bits.append(8)
            elif p.symbol() == "N":
                bits.append(9)
            elif p.symbol() == "B":
                bits.append(10)
            elif p.symbol() == "Q":
                bits.append(11)
            elif p.symbol() == "K":
                bits.append(12)
            elif p.symbol() == "p":
                bits.append(1)
            elif p.symbol() == "r":
                bits.append(2)
            elif p.symbol() == "n":
                bits.append(3)
            elif p.symbol() == "b":
                bits.append(4)
            elif p.symbol() == "q":
                bits.append(5)
            elif p.symbol() == "k":
                bits.append(6)
    if board.turn == chess.WHITE:
        bits.append(13)
        other = chess.BLACK
    else:
        bits.append(14)
        other = chess.WHITE
        # does player have castling rights
    if board.has_castling_rights(board.turn):
        bits.append(15)
    else:
        bits.append(16)
        # does opponent have castling rights
    if board.has_castling_rights(other):
        bits.append(15)
    else:
        bits.append(16)
        # is player in check
    if board.is_check():
        bits.append(17)
    else:
        bits.append(18)
    #bits.append(19)
    return bits

class WinPred(Dataset):
    def __init__(self, filepath):
        self.device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.data = pd.read_csv(f"{filepath}.csv")
        print(f"set has {len(self.data.index)} data points")

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, item):
        pos = self.data['position'].iloc[item]
        board_idx = board_encoder(pos)
        x = torch.tensor(board_idx + [19], dtype=torch.long, device=self.device)
        if self.data["result"] == "win":
            y = 1
        elif self.data["result"] == "draw":
            y = 0.5
        elif self.data["result"] == "lose":
            y = 0
        return x, y


def init_chessGPT(t):
    # create model instance
    model_config = GPT.get_default_config()
    model_config.model_type = t
    model_config.vocab_size = 15
    model_config.block_size = 65
    model = GPT(model_config)
    return model


def show_losses(dict):
    losses = [x.cpu().detach() for x in dict["loss"]]
    print("avg. first ten ", sum(losses[0:10])/10)
    print("avg. loss", sum(losses)/len(losses))
    print("avg. last ten ", sum(losses[-10:])/10)
    plt.plot(dict["time"], losses)
    plt.suptitle("Losses During Training")
    plt.xlabel("Iters")
    plt.ylabel("Loss")
    plt.show()
    return


if __name__ == '__main__':
    trainfile = ""
    testfile = ""
    modelfile = ""
    traindata = WinPred(trainfile)
    testdata = WinPred(testfile)
    model = init_chessGPT("ChessGPTMed")
    trainer = WinTrainer.get_default_config(batch=512)
    trainer.max_iters = 200000
    model, clf, losses = WinTrainer(trainer, model, traindata, testdata).run()
    show_losses(losses)
    # save
    state_dict = model.state_dict()
    torch.save(state_dict, f"{modelfile}Encoder.pt")
    state_dict = clf.state_dict()
    torch.save(state_dict, f"{modelfile}Classifier.pt")
  
