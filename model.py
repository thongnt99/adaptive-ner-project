import torch
import torch 
import torch.nn as nn 
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from flair.embeddings import FlairEmbeddings, BertEmbeddings
from flair.embeddings import StackedEmbeddings
from flair.data import Sentence
START_TAG = "<START>"
STOP_TAG = "<STOP>"
UNK = "<UNK>"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()

def log_sum_exp(vec, dim):
    max_value, idx = torch.max(vec, dim)
    max_exp = max_value.unsqueeze(-1).expand_as(vec)
    return max_value + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, use_transformer = False, ix_to_word = None):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.use_transformer = use_transformer
        if self.use_transformer:
            self.flair_forward_embedding = FlairEmbeddings('multi-forward')
            self.flair_backward_embedding = FlairEmbeddings('multi-backward')
            self.bert_embedding = BertEmbeddings('bert-base-cased')
            self.stacked_embeddings = StackedEmbeddings(embeddings=[self.flair_forward_embedding, self.flair_backward_embedding, self.bert_embedding])
            self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers = 1, bidirectional =True, batch_first=True, dropout= 0.005)
        else: 
            self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim//2, num_layers = 1, bidirectional =True, batch_first=True, dropout= 0.005)
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000
        self.ix_to_word = ix_to_word
        # self.hidden = self.init_hidden()
        
    def init_hidden(self):
        return (torch.zeros(2,self.batch_size, self.hidden_dim //2).to(device),torch.zeros(2, self.batch_size, self.hidden_dim //2).to(device))
    
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
        
    def get_flair_embedding(self, sentences, lens):
        ## convert id back to words
        text_sentences = []
        for (sent,sl) in zip(sentences,lens):
            no_paddings = sent[:sl]
            word_list = [self.ix_to_word[ix.item()] for ix in no_paddings]
            text_sentences.append(" ".join(word_list))
        embeddings_tensor = torch.zeros(sentences.size(0), sentences.size(1),  self.embedding_dim)
        for i,text_sent in enumerate(text_sentences):
            flair_sent = Sentence(text_sent)
            self.bert_embedding.embed(flair_sent)
            # print(text_sent)
            for j, word in enumerate(flair_sent):
                embeddings_tensor[i,j] = word.embedding
        embeddings_tensor = embeddings_tensor.to(device)
        return embeddings_tensor

    def _get_lstm_features(self, sentence, lens):
        self.batch_size,_ = sentence.size()
        self.hidden = self.init_hidden()
        if self.use_transformer:
            embeds = self.get_flair_embedding(sentence, lens)
        else:
            embeds = self.word_embeds(sentence)
        embeds = pack_padded_sequence(embeds, lens.tolist(), batch_first= True)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags, lens):
        self.batch_size,_,_ = feats.size()
        score  = torch.zeros(self.batch_size,1).to(device)
        feats_t = feats.transpose(0,1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype = torch.long).to(device).unsqueeze(0).expand(self.batch_size,1), tags],1)
        tags_t = tags.transpose(0,1)
        c_lens = lens.clone()
        for i, feat in enumerate(feats_t):
            lens_mask = (c_lens > 0).float().unsqueeze(-1)
            score = score + lens_mask*self.transitions[tags_t[i+1], tags_t[i]].unsqueeze(-1) + torch.gather(feat,1,tags_t[i+1].unsqueeze(-1))
            lens_mask = (c_lens == 1).float().unsqueeze(-1)
            score = score + lens_mask*self.transitions[self.tag_to_ix[STOP_TAG], tags_t[i+1]].unsqueeze(-1)
            c_lens = c_lens - 1
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
        feats = self._get_lstm_features(sentences,lens)
        forward_score = self._forward_alg(feats,lens)
        gold_score = self._score_sentence(feats, sent2tags, lens)
        return (forward_score - gold_score).mean()
    
    def forward(self, sentence, lens):
        self.batch_size,_ = sentence.size()
        lstm_feats = self._get_lstm_features(sentence,lens)
        scores, paths = self._viterbi_decode(lstm_feats, lens)
        return scores, paths

