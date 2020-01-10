import numpy as np
import torch 
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertForTokenClassification, BertAdam
from keras.preprocessing.sequence import pad_sequences
from seqeval.metrics import f1_score
from tqdm import trange

def read_data(file):
  sentences = []
  with open(file, "r") as f:
    for line in f.readlines():
      sentences.append([word for word in line.strip().split(" ")])
  return sentences

def tokenize_data(input_sentence_list, input_tags_list, tokenizer):
  output_sentence_list = []
  output_tags_list = []
  for words, tags in zip(input_sentence_list, input_tags_list):
    bert_tokens = []
    bert_tags = []
    for word, tag in zip(words, tags):
      tokens_of_word = tokenizer.tokenize(word)
      bert_tokens.extend(tokens_of_word)
      bert_tags.extend([tag]*len(tokens_of_word))
    output_sentence_list.append(bert_tokens)
    output_tags_list.append(bert_tags)
  return output_sentence_list, output_tags_list

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=2).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)
if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  n_gpu = torch.cuda.device_count()
  MAX_LEN = 75
  bs = 32
  train_text_path = "data/train/sentences.txt"
  train_label_path = "data/train/labels.txt"
  test_text_path = "data/test/sentences.txt"
  test_label_path = "data/test/labels.txt"
  val_text_path = "data/val/sentences.txt"
  val_label_path = "data/val/labels.txt"
  train_sents = read_data(train_text_path)
  train_tags = read_data(train_label_path)
  test_sents = read_data(test_text_path)
  test_tags = read_data(test_label_path)
  val_sents = read_data(val_text_path)
  val_tags = read_data(val_label_path)
  label2id = {}
  id2label = {}
  with open("tags.txt", "r") as f:
      for idx, label in enumerate(f.readlines()):
          label2id[label.strip()] = idx 
          id2label[idx] = label 
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
  tokenized_train_text, tokenized_train_tags = tokenize_data(train_sents, train_tags, tokenizer)
  tokenized_test_text, tokenized_test_tags = tokenize_data(test_sents, test_tags, tokenizer)
  tokenized_val_text, tokenized_val_tags = tokenize_data(val_sents, val_tags, tokenizer)
  train_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_train_text],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  train_output_ids = pad_sequences([[label2id.get(l) for l in lab] for lab in tokenized_train_tags],
                      maxlen=MAX_LEN, value=label2id["O"], padding="post",
                      dtype="long", truncating="post")
  test_input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_test_text],
                            maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
  test_output_ids = pad_sequences([[label2id.get(l) for l in lab] for lab in tokenized_test_tags],
                      maxlen=MAX_LEN, value=label2id["O"], padding="post",
                      dtype="long", truncating="post")
  train_attention_masks = [[float(i>0) for i in ii] for ii in train_input_ids]
  test_attention_masks = [[float(i>0) for i in ii] for ii in test_input_ids]
  train_x_tensor = torch.tensor(train_input_ids).to(device)
  train_y_tensor = torch.tensor(train_output_ids).to(device)
  train_mask_tensor = torch.tensor(train_attention_masks).to(device)
  test_x_tensor = torch.tensor(test_input_ids).to(device)
  test_y_tensor = torch.tensor(test_output_ids).to(device)
  test_mask_tensor = torch.tensor(test_attention_masks).to(device)

  train_data = TensorDataset(train_x_tensor, train_mask_tensor, train_y_tensor)
  train_sampler = RandomSampler(train_data)
  train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

  test_data = TensorDataset(test_x_tensor, test_mask_tensor, test_y_tensor)
  test_sampler = SequentialSampler(test_data)
  test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=bs)
  model = BertForTokenClassification.from_pretrained("bert-base-uncased", num_labels=len(label2id)).to(device)
  FULL_FINETUNING = False 
  if FULL_FINETUNING:
      param_optimizer = list(model.named_parameters())
      no_decay = ['bias', 'gamma', 'beta']
      optimizer_grouped_parameters = [
          {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.01},
          {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
          'weight_decay_rate': 0.0}
      ]
  else:
      param_optimizer = list(model.classifier.named_parameters()) 
      optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]
  optimizer = Adam(optimizer_grouped_parameters, lr=3e-5)
  epochs = 5
  max_grad_norm = 1.0

  for _ in trange(epochs, desc="Epoch"):
    # TRAIN loop
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # forward pass
        loss = model(b_input_ids, token_type_ids=None,
                      attention_mask=b_input_mask, labels=b_labels)
        # backward pass
        loss.backward()
        # track train loss
        tr_loss += loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        model.zero_grad()
    # print train loss per epoch
    print("Train loss: {}".format(tr_loss/nb_tr_steps))
    # VALIDATION on validation set
    model.eval()
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        with torch.no_grad():
            tmp_eval_loss = model(b_input_ids, token_type_ids=None,
                                  attention_mask=b_input_mask, labels=b_labels)
            logits = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.append(label_ids)
        
        tmp_eval_accuracy = flat_accuracy(logits, label_ids)
        
        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy
        
        nb_eval_examples += b_input_ids.size(0)
        nb_eval_steps += 1
    eval_loss = eval_loss/nb_eval_steps
    print("Validation loss: {}".format(eval_loss))
    print("Validation Accuracy: {}".format(eval_accuracy/nb_eval_steps))
    pred_tags = [id2label[p_i] for p in predictions for p_i in p]
    valid_tags = [id2label[l_ii] for l in true_labels for l_i in l for l_ii in l_i]
    print("F1-Score: {}".format(f1_score(pred_tags, valid_tags)))