import os.path
import torch 
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from prepare_data import build_vocab
from prepare_data import load_word_vocab
from prepare_data import load_lab_vocab
from prepare_data import get_data_loader 
from seqeval.metrics import classification_report
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score
from model import BiLSTM_CRF

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EMBEDDING_DIM = 300
HIDDEN_DIM = 400
epochs = 20
BS = 64

def seqid2lab(id_seq, ix_to_lab):
    seq = [ix_to_lab[id.item()] for id in id_seq]
    return seq

if __name__ == "__main__":
    train_folder = "data/train"
    val_folder = "data/val"
    test_folder = "data/test"
    if not os.path.isfile("resources/vocab.txt") and not os.path.isfile("resources/labs.txt"):
        print("building word and label vocabulary")
        build_vocab([train_folder, val_folder, test_folder])
    word_to_ix, ix_to_word = load_word_vocab()
    assert len(word_to_ix) == len(ix_to_word)
    lab_to_ix, ix_to_lab = load_lab_vocab()
    assert len(lab_to_ix) == len(ix_to_lab)
    train_data_loader = get_data_loader(train_folder, word_to_ix, lab_to_ix)
    test_data_loader = get_data_loader(test_folder, word_to_ix, lab_to_ix)
    val_data_loader = get_data_loader(val_folder, word_to_ix, lab_to_ix)

    model = BiLSTM_CRF(len(word_to_ix), lab_to_ix, EMBEDDING_DIM, HIDDEN_DIM).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_f1 = -1
    for epoch in range(epochs): 
        for i, batch in enumerate(train_data_loader):
            model.zero_grad()
            sents, labs, lens = batch
            sents = pad_sequence(sents,batch_first= True).to(device)
            labs = pad_sequence(labs,batch_first= True).to(device)
            lens = torch.tensor(lens).to(device)
            lens, idx  = torch.sort(lens, descending= True)
            sents = sents[idx]
            labs = labs[idx]
            loss = model.neg_log_likelihood(sents, labs, lens)
            loss.backward()
            optimizer.step()
            score, preds = model(sents, lens)
            true_labs = [seqid2lab(labs[i,:l], ix_to_lab) for i,l in enumerate(lens)]
            pred_labs = [seqid2lab(preds[i,:l], ix_to_lab) for i,l in enumerate(lens)]
            acc = accuracy_score(true_labs, pred_labs)
            f1 = f1_score(true_labs, pred_labs)
            print("Epoch {}, batch {}, train loss {:.4f}, train acc {:.4f}, train f1 {:.4f} ".format(epoch, i, loss.item(), acc, f1))
            if ((i+1)%50 == 0):
                with torch.no_grad():
                    model.eval()
                    print("Evaluation on validation set")
                    true_labels = []
                    pred_labels = []
                    for batch in val_data_loader:
                        sents, labs, lens = batch
                        sents = pad_sequence(sents,batch_first= True).to(device)
                        labs = pad_sequence(labs,batch_first= True).to(device)
                        lens = torch.tensor(lens).to(device)
                        lens, idx = torch.sort(lens, descending= True)
                        sents = sents[idx]
                        labs = labs[idx]
                        score, preds = model(sents, lens)
                        for i, l in enumerate(lens):
                            true_labels.append(seqid2lab(labs[i,:l],ix_to_lab))
                            pred_labels.append(seqid2lab(preds[i,:l],ix_to_lab))
                    f1= f1_score(true_labels, pred_labels)
                    if (f1 > best_f1):
                        torch.save(model.state_dict(), "models/model-27-02-20")
                        best_f1 = f1

                    print("Accuracy: {:.4f}".format(accuracy_score(true_labels, pred_labels)))
                    print("F1 score: {:.4f}".format(f1))
                    print(classification_report(true_labels, pred_labels))
                    model.train(True)
    with torch.no_grad():
        print("Evaluation on test set")
        model.load_state_dict(torch.load("models/model-27-20-20"))
        model.eval()
        true_labels = []
        pred_labels = []
        for batch in test_data_loader:
            sents, labs, lens = batch
            sents = pad_sequence(sents,batch_first= True).to(device)
            labs = pad_sequence(labs,batch_first= True).to(device)
            lens = torch.tensor(lens).to(device)
            lens, idx = torch.sort(lens, descending= True)
            sents = sents[idx]
            labs = labs[idx]
            score, preds = model(sents, lens)
            for i, l in enumerate(lens):
                true_labels.append(seqid2lab(labs[i,:l],ix_to_lab))
                pred_labels.append(seqid2lab(preds[i,:l],ix_to_lab))
        print("Accuracy: {:.4f}".format(accuracy_score(true_labels, pred_labels)))
        print("F1 score: {:.4f}".format(f1))
        print(classification_report(true_labels, pred_labels))
    