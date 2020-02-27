from collections import Counter
import torch 
from torch.utils.data import Dataset, DataLoader
from model import UNK
from model import START_TAG
from model import STOP_TAG

class NERDataset(Dataset):
    
    def __init__(self, texts, labels, lens):
        self.texts = texts 
        self.labels = labels 
        self.lens = lens
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return self.texts[index], self.labels[index], self.lens[index]


def prepare_sequence(seq, to_ix):
    """
    Convert word (label) sequence to id sequencec
    """
    idxs = [to_ix[w] if w in to_ix else to_ix[UNK] for w in seq]
    return torch.tensor(idxs, dtype = torch.long)

def read_data(data_path):
    """
    Read data from a folder, the folder should contains "sentences.txt" and "labels.txt"
    """
    text_seqs = []
    lab_seqs = []
    with open(data_path + "/sentences.txt", "r") as f:
        for line in f.readlines():
            text_seqs.append(line.strip().split())
    with open(data_path+"/labels.txt", "r") as f:
        for line in f.readlines():
            lab_seqs.append(line.strip().split())
    return text_seqs, lab_seqs

def save_vocab(counter, path):
    with open(path, "w") as f:
        for pair in counter:
            f.write("{}\t{}\n".format(pair[0], pair[1]))

def load_dict(path):
    word_to_ix = {}
    ix_to_word = {}
    with open(path) as f:
        for ix,line in enumerate(f.readlines()):
            word = line.split("\t")[0]
            word_to_ix[word] = ix 
            ix_to_word[ix] = word
    return word_to_ix, ix_to_word

def load_word_vocab():
    word_to_ix, ix_to_word = load_dict("resources/vocab.txt")
    ## Add oov token to vocab 
    word_to_ix[UNK] = len(word_to_ix)
    ix_to_word[len(word_to_ix)-1] = UNK
    return word_to_ix, ix_to_word
def load_lab_vocab():
    lab_to_ix, ix_to_lab = load_dict("resources/labs.txt")
    ## Add BEGIN and STOP tags 
    lab_to_ix[START_TAG] = len(lab_to_ix)
    ix_to_lab[len(lab_to_ix) -1] = START_TAG
    lab_to_ix[STOP_TAG] = len(lab_to_ix)
    ix_to_lab[len(lab_to_ix) -1] = STOP_TAG
    return lab_to_ix, ix_to_lab

def build_vocab(dir_list):
    """
    Build word and label vocabulary from data 
    """
    word_counter = Counter()
    lab_counter = Counter()
    for d in dir_list:
        text_seqs, lab_seqs = read_data(d)
        for seq in text_seqs:
            for word in seq: 
                word_counter[word] +=1
        for seq in lab_seqs:
            for lab in seq:
                lab_counter[lab] +=1
    word_vocab = word_counter.most_common()
    lab_vocab = lab_counter.most_common()
    save_vocab(word_vocab, "resources/vocab.txt")
    save_vocab(lab_vocab, "resources/labs.txt")
    return word_vocab, lab_vocab



def get_data_set(dir, word_to_ix, lab_to_ix):
    """
    Get data loader for train/test/val
    """
    text_seqs, lab_seqs = read_data(dir)
    text_id_seqs  = [prepare_sequence(seq, word_to_ix) for seq in text_seqs]
    lab_id_seqs = [prepare_sequence(seq, lab_to_ix) for seq in lab_seqs]
    lens = [len(text) for text in text_seqs]
    dataset = NERDataset(text_id_seqs, lab_id_seqs, lens)
    return dataset

def my_collate(batch):
    """
    Return data for each batch 
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    lens = [item[2] for item in batch]
    return [data, target, lens]

params = {
   "batch_size": 32, 
    "shuffle": False,
    "collate_fn": my_collate
}

def get_data_loader(dir,  word_to_ix, lab_to_ix, params = params):
    """
    Return dataloader iterating thought batches 
    """
    dataset = get_data_set(dir, word_to_ix, lab_to_ix)
    return DataLoader(dataset, **params)
