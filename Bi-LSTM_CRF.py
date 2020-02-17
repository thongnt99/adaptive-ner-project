#!/usr/bin/env python
# coding: utf-8

# In[135]:


import torch 
import torch.autograd as autograd
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
torch.manual_seed(1)


# In[136]:


def argmax(vec):
    # return the argmax as a python int 
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] if w in to_ix else to_ix[UNK] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

def log_sum_exp(vec, dim):
    max_value, idx = torch.max(vec, dim)
    max_exp = max_value.unsqueeze(-1).expand_as(vec)
    return max_value + torch.log(torch.sum(torch.exp(vec - max_exp), dim))


# In[164]:


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, batch_size):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers = 1, bidirectional =True, batch_first=True)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.batch_size = batch_size
        self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.randn(2,self.batch_size, self.hidden_dim //2).to(device),torch.randn(2, self.batch_size, self.hidden_dim //2).to(device))
    
    def _forward_alg(self, feats, lens):
        self.batch_size, _, _ = feats.size()
        alphas = torch.full((self.batch_size, self.tagset_size), -10000.).to(device)
        alphas[:, self.tag_to_ix[START_TAG]] = 0 
        feats_t = feats.transpose(1,0)
        c_lens = lens.clone()
        for feat in feats_t:
            feat_exp = feat.unsqueeze(-1).expand(self.batch_size, self.tagset_size, self.tagset_size)
            alpha_exp = alphas.unsqueeze(1).expand(self.batch_size, self.tagset_size, self.tagset_size)
            trans_exp = self.transitions.unsqueeze(0).expand(self.batch_size, self.tagset_size, self.tagset_size)
            mat = trans_exp + alpha_exp + feat_exp 
            next_alpha = log_sum_exp(mat, 2).squeeze(-1)
            lens_mask = (c_lens > 0).float().unsqueeze(-1).expand(self.batch_size, self.tagset_size)
            alphas = lens_mask*next_alpha + (1-lens_mask)*alphas
            c_lens = c_lens - 1
        alphas = alphas + self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0).expand_as(alphas)
        return alphas 
    
    def _get_lstm_features(self, sentence):
        self.batch_size,_ = sentence.size()
        self.hidden = self.init_hidden()
        embeds = self.word_embeds(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags, lens):
        self.batch_size,_,_ = feats.size()
        score  = torch.zeros(self.batch_size,1).to(device)
        feats_t = feats.transpose(0,1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype = torch.long).unsqueeze(0).expand(self.batch_size,1), tags],1)
        tags_t = tags.transpose(0,1)
        c_lens = lens.clone()
        for i, feat in enumerate(feats_t):
            lens_mask = (c_lens > 0).float().unsqueeze(-1)
            score = score + lens_mask*self.transitions[tags_t[i+1], tags_t[i]].unsqueeze(-1) + torch.gather(feat,1,tags_t[i+1].unsqueeze(-1))
            lens_mask = (c_lens == 1).float().unsqueeze(-1)
            score = score + lens_mask*self.transitions[self.tag_to_ix[STOP_TAG], tags_t[i+1]].unsqueeze(-1)
        return score 
    
    def _viterbi_decode(self, feats, lens):
        self.batch_size,_,_ = feats.size()
        backpointers = [] 
        pre_selection = torch.full((self.batch_size, self.tagset_size), -10000.).to(device)
        pre_selection[:,self.tag_to_ix[START_TAG]] = 0
        feats_t = feats.transpose(1,0)
        c_lens = lens.clone()
        
        for feat in feats_t:
            feat_exp = feat.unsqueeze(-1).expand(self.batch_size, self.tagset_size, self.tagset_size)
            pre_exp = pre_selection.unsqueeze(1).expand(self.batch_size, self.tagset_size, self.tagset_size)
            tran_exp = self.transitions.unsqueeze(0).expand(self.batch_size, self.tagset_size, self.tagset_size)
            mat = feat_exp + pre_exp + tran_exp
            
            next_selection, next_idx = torch.max(mat, 2)
            next_selection = next_selection.squeeze(-1)
            next_idx = next_idx.squeeze(-1).unsqueeze(0)
            
            lens_mask = (c_lens > 0).float().unsqueeze(-1).expand(self.batch_size, self.tagset_size)
            pre_selection = lens_mask*next_selection + (1-lens_mask)*pre_selection
            backpointers.append(next_idx)
            
            lens_mask = (c_lens == 1).float().unsqueeze(-1).expand(self.batch_size, self.tagset_size)
            pre_selection = pre_selection + lens_mask*self.transitions[self.tag_to_ix[STOP_TAG]].unsqueeze(0).expand(self.batch_size, self.tagset_size)
            c_lens = c_lens -1 
        scores, idx = torch.max(pre_selection,1)
        idx = idx.squeeze(-1)
        backpointers = torch.cat(backpointers)
        paths = [idx.unsqueeze(1)]
        for argmax in reversed(backpointers):
            idx_exp = idx.unsqueeze(-1)

            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)
            paths.insert(0, idx.unsqueeze(1))
        paths = torch.cat(paths[1:],1)
        scores = scores.squeeze(-1)
        return scores, paths
    
    def neg_log_likelihood(self, sentences, sent2tags, lens):
        self.batch_size, _ = sentences.size()
        feats = self._get_lstm_features(sentences)
        forward_score = self._forward_alg(feats,lens)
        gold_score = self._score_sentence(feats, sent2tags, lens)
        return (forward_score - gold_score).mean()
    
    def forward(self, sentence, lens):
        self.batch_size,_ = sentence.size()
        lstm_feats = self._get_lstm_features(sentence)
        scores, paths = self._viterbi_decode(lstm_feats, lens)
        return scores, paths


# #### Training 

# In[148]:


START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
EMBEDDING_DIM = 300
HIDDEN_DIM = 600
epochs = 50
BS = 64


# In[139]:


def read_data(data_path):
    text_seqs = []
    lab_seqs = []
    with open(data_path + "/sentences.txt", "r") as f:
        for line in f.readlines():
            text_seqs.append(line.strip().split())
    with open(data_path+"/labels.txt", "r") as f:
        for line in f.readlines():
            lab_seqs.append(line.strip().split())
    for i in range(len(text_seqs)):
        if (len(text_seqs[i]) != len(lab_seqs[i])):
            print(data_path, " ", i," ", len(text_seqs[i]), " ", len(lab_seqs[i]))
    return text_seqs, lab_seqs
train_folder = "data/train"
val_folder = "data/val"
test_folder = "data/test"
text_seqs_train, lab_seqs_train = read_data(train_folder)
text_seqs_val, lab_seqs_val = read_data(val_folder)
text_seqs_test, lab_seqs_test = read_data(test_folder)


# In[140]:


# building lexicon and tagset
word_to_ix = {}
tag_to_ix = {}
ix_to_tag = {}
for seq in text_seqs_train:
    for word in seq:
        if not word in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
for seq in lab_seqs_train:
    for lab in seq:
        if not lab in tag_to_ix:
            idx = len(tag_to_ix)
            tag_to_ix[lab] = idx
            ix_to_tag[idx] = lab
word_to_ix[UNK] = len(word_to_ix)
idx = len(tag_to_ix)
tag_to_ix[START_TAG] = idx 
ix_to_tag[idx] = START_TAG
idx = len(tag_to_ix)
tag_to_ix[STOP_TAG] = idx
ix_to_tag[idx] = START_TAG


# In[141]:


def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    lens = [item[2] for item in batch]
    return [data, target, lens]


# In[142]:


class NERDataset(Dataset):
    
    def __init__(self, texts, labels, lens):
        self.texts = texts 
        self.labels = labels 
        self.lens = lens
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index], self.lens[index]


# In[143]:


# Making data loader 
params = {
   "batch_size": BS, 
    "shuffle": True,
    "collate_fn": my_collate
}
train_text_ids  = [prepare_sequence(seq, word_to_ix) for seq in text_seqs_train]
train_label_ids = [prepare_sequence(seq, tag_to_ix) for seq in lab_seqs_train]
train_lens = [len(text) for text in train_text_ids]
train_dataset = NERDataset(train_text_ids, train_label_ids, train_lens)
train_dataloader = DataLoader(train_dataset, **params)

test_text_ids  = [prepare_sequence(seq, word_to_ix) for seq in text_seqs_test]
test_label_ids = [prepare_sequence(seq, tag_to_ix) for seq in lab_seqs_test]
test_lens = [len(text) for text in test_text_ids]
test_dataset = NERDataset(test_text_ids, test_label_ids, test_lens)
test_dataloader = DataLoader(test_dataset, **params)

val_text_ids  = [prepare_sequence(seq, word_to_ix) for seq in text_seqs_val]
val_label_ids = [prepare_sequence(seq, tag_to_ix) for seq in lab_seqs_val]
val_lens = [len(text) for text in val_text_ids]
val_dataset = NERDataset(val_text_ids, test_label_ids, val_lens)
val_dataloader = DataLoader(val_dataset, **params)


# In[144]:


def id2lab(id_seq):
    seq = [ix_to_tag[id.item()] for id in id_seq]
    return seq


# In[161]:


from torch.nn.utils.rnn import pad_sequence
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
from seqeval.metrics import classification_report
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM, BS).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.001, weight_decay=1e-4)

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    for i, batch in enumerate(train_dataloader):
        model.zero_grad()
        sents, labs, lens = batch
        sents = pad_sequence(sents,batch_first= True).to(device)
        labs = pad_sequence(labs,batch_first= True).to(device)
        lens = torch.tensor(lens).to(device)
        loss = model.neg_log_likelihood(sents, labs, lens)
        loss.backward()
        optimizer.step()
        if (i%100 == 0):
            print("Epoch {}, batch {}, loss {}".format(epoch, i, loss.item()))
            with torch.no_grad():
                    true_labels = []
                    pred_labels = []
                    for batch in test_dataloader:
                        sents, labs, lens = batch
                        sents = pad_sequence(sents,batch_first= True).to(device)
                        labs = pad_sequence(labs,batch_first= True).to(device)
                        lens = torch.tensor(lens).to(device)
                        score, preds = model(sents, lens)
        #                         print(sents.size()," ",labs.size(),"  ",preds.size())
                        for i, l in enumerate(lens):
                            true_labels.append(id2lab(labs[i,:l]))
                            pred_labels.append(id2lab(preds[i,:l]))
        #                             print(true_labels)
        #                             print(pred_labels)
                    print(classification_report(true_labels, pred_labels))

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))

