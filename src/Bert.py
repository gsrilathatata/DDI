import os
import matplotlib
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertConfig
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup, Tag
from tqdm import tqdm, trange
import transformers
from transformers import BertForTokenClassification, AdamW
import Postprocssing
from SentenceGetter import SentenceGetter
from transformers import get_linear_schedule_with_warmup
from seqeval.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from xml.etree.ElementTree import ElementTree,Element
import sys
## Sample code is picked from this website and some customization is done according to the need.
##https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
## I would like to thank the author for thier contribution in the complex concept implementation
## the code is used carefully and only required customisations are done and credits to the authos
################################################################################
## The main module for implementing the BERT algorithm                        ##
## In this module pre-trained model is loaded and updated with custom tagged  ## 
## training dataset, and evaluated for loss and fine tuning.                  ##
## Finally, the model is used for testing the untagged sentences.             ##
################################################################################


torch.__version__
transformers.__version__
MAX_LEN = 128
bs = 32
FULL_FINETUNING = True
epochs = 2
max_grad_norm = 1.0

if len(sys.argv) < 3:
    print("Arguments required -  trigger or si and test or eval")
    exit(0)

typeofops = sys.argv[1]
test_eval = sys.argv[2]
################################################################################
##                               Configuration Section                        ##
##       In this section, the the required folder structure is defined        ## 
################################################################################

if test_eval == 'test':
   if typeofops == 'trigger':
      path_test =  "../data/test/test"
      path_guess = "../data/test/guess"
      #path_gold = "GOLD"
      path_pretty = "../data/test/prettify"
      data = pd.read_csv("../data/test/trainin_csv/train.csv", encoding="latin1").fillna(method="ffill")
      trigger_si = "Precipitant_Trigger"
   elif typeofops == 'si':
      path_test = "../data/test/guess"
      path_guess = "../data/test/guess"
      #path_gold = "GOLD"
      path_pretty = "../data/test/prettify"
      data = pd.read_csv("../data/test/trainin_csv/train.csv", encoding="latin1").fillna(method="ffill")
      trigger_si = "SpecificInteractions"
elif test_eval == 'eval':
   if typeofops == 'trigger':
      path_test = "../data/eval/test"
      path_guess = "../data/eval/guess"
      path_gold = "../data/eval/gold"
      path_pretty = "../data/eval/prettify"
      data = pd.read_csv("../data/test/trainin_csv/train.csv", encoding="latin1").fillna(method="ffill")
      Postprocssing.removeMention(path_gold,path_test)
      trigger_si = "Precipitant_Trigger"
   elif typeofops == 'si':
      path_test = "../data/eval/test_si"
      path_guess = "../data/eval/guess_si"
      path_gold = "../data/eval/gold"
      path_pretty = "../data/eval/prettify_si"
      Postprocssing.removeMention(path_gold,path_test)
      #data = pd.read_csv("../data/eval/trainin_csv/train_si.csv", encoding="latin1").fillna(method="ffill")
      data = pd.read_csv("../data/test/trainin_csv/train_si.csv", encoding="latin1").fillna(method="ffill")
      trigger_si = "SpecificInteractions"
################################################################################
##Sentence aggregation based on the sentence number from the training dataset ##
################################################################################

getter = SentenceGetter(data)
sentences = [[word[0] for word in sentence] for sentence in getter.sentences]
labels = [[s[1] for s in sentence] for sentence in getter.sentences]
tag_values = list(set(data["tag"].values))
tag_values.append("PAD")
tag2idx = {t: i for i, t in enumerate(tag_values)}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
torch.cuda.get_device_name(0)

################################################################################
##         Tokenise the sentences using BERT Tokeniser                        ##
################################################################################
tokenizer = BertTokenizer.from_pretrained('bert-large-cased', do_lower_case=False)
tokenized_texts_and_labels = [
    Postprocssing.tokenize_and_preserve_labels(sent, labs,tokenizer)
    for sent, labs in zip(sentences, labels)
]

################################################################################
##         Tokenise the sentences using BERT Tokeniser                        ##
################################################################################
tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                          maxlen=MAX_LEN, dtype="int64", value=0.0,
                          truncating="post", padding="post")

tags = pad_sequences([[tag2idx.get(l) for l in lab] for lab in labels],
                     maxlen=MAX_LEN, value=tag2idx["PAD"], padding="post",
                     dtype="int64", truncating="post")

attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags,
                                                            random_state=2018, test_size=0.1)
tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids,
                                             random_state=2018, test_size=0.1)

################################################################################
##         Convert th input tokens and labels to tensor matrices              ##
################################################################################
tr_inputs = torch.tensor(tr_inputs)
val_inputs = torch.tensor(val_inputs)
tr_tags = torch.tensor(tr_tags)
val_tags = torch.tensor(val_tags)
tr_masks = torch.tensor(tr_masks)
val_masks = torch.tensor(val_masks)

train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
train_sampler = RandomSampler(train_data)
train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=bs)

valid_data = TensorDataset(val_inputs, val_masks, val_tags)
valid_sampler = SequentialSampler(valid_data)
valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=bs)

################################################################################
##         Set the pretrained model paramaters                                ##
################################################################################
model = BertForTokenClassification.from_pretrained(
    "bert-base-cased",
    num_labels=len(tag2idx),
    output_attentions = False,
    output_hidden_states = False
)

model.cuda();
################################################################################
##         Set the optimiser parameter                                        ##
################################################################################
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

################################################################################
##      Initilaise the custom parameters of the optimiser like learning rate  ##
##         type of optimiser etc.                                             ##
################################################################################
optimizer = AdamW(
    optimizer_grouped_parameters,
    lr=3e-5,
    eps=1e-8
)

# Total number of training steps is number of batches * number of epochs.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

## Store the average loss after each epoch so we can plot them.
loss_values, validation_loss_values = [], []

for _ in trange(epochs, desc="Epoch"):
    model.train()
    total_loss = 0
    # Training loop
    for step, batch in enumerate(train_dataloader):
        # add batch to gpu
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        # Always clear any previously calculated gradients before performing a backward pass.
        model.zero_grad()
        # forward pass
        # This will return the loss (rather than the model output)
        # because we have provided the `labels`.
        outputs = model(b_input_ids, token_type_ids=None,
                        attention_mask=b_input_mask, labels=b_labels)
        # get the loss
        loss = outputs[0]
        # Perform a backward pass to calculate the gradients.
        loss.backward()
        # track train loss
        total_loss += loss.item()
        # Clip the norm of the gradient
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
        # update parameters
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
    # Calculate the average loss over the training data.
    avg_train_loss = total_loss / len(train_dataloader)
    print("Average train loss: {}".format(avg_train_loss))

    # Store the loss value for plotting the learning curve.
    loss_values.append(avg_train_loss)

    # Put the model into evaluation mode`
    model.eval()
    # Reset the validation loss for this epoch.
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    predictions , true_labels = [], []
    for batch in valid_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch

        # Telling the model not to compute or store gradients,
        # saving memory and speeding up validation
        with torch.no_grad():
            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have not provided labels.
            outputs = model(b_input_ids, token_type_ids=None,
                            attention_mask=b_input_mask, labels=b_labels)
        # Move logits and labels to CPU
        logits = outputs[1].detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Calculate the accuracy for this batch of test sentences.
        eval_loss += outputs[0].mean().item()
        predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
        true_labels.extend(label_ids)

    eval_loss = eval_loss / len(valid_dataloader)
    validation_loss_values.append(eval_loss)
    print("Validation loss: {}".format(eval_loss))
    pred_tags = [tag_values[p_i] for p, l in zip(predictions, true_labels)
                                 for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
    valid_tags = [tag_values[l_i] for l in true_labels
                                  for l_i in l if tag_values[l_i] != "PAD"]
    print("Validation Accuracy: {}".format(accuracy_score(pred_tags, valid_tags)))
    print("Validation F1-Score: {}".format(f1_score(pred_tags, valid_tags)))
    print()

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(loss_values, 'b-o', label="training loss")
    plt.plot(validation_loss_values, 'r-o', label="validation loss")

    # Label the plot.
    plt.title("Learning curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

   # plt.show()

################################################################################
##         prediction of the model starts here                                ##
################################################################################
##########################
countforid = 0
docs = []
for file in os.listdir(path_test):
    filepath = os.path.join(path_test, file)
    filepath1 = os.path.join(path_pretty, file)
    tree = ElementTree()
    tree.parse(filepath)

    sentences = tree.findall('.//Sentence')
    for elem in sentences:
            for child in elem:
                if type(child) == Element:
                    if child.tag == "SentenceText":
                      test_sentence = (child.text).strip()

                      tokenized_sentence = tokenizer.encode(test_sentence)

                      #input_ids = torch.tensor([tokenized_sentence])
                      input_ids = torch.tensor([tokenized_sentence]).cuda()

                      with torch.no_grad():
                          output = model(input_ids)
                      label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)

                      # join bpe split tokens
                      tokens = tokenizer.convert_ids_to_tokens(input_ids.to('cpu').numpy()[0])
                      new_tokens, new_labels = [], []
                      for token, label_idx in zip(tokens, label_indices[0]):
                          if token.startswith("##"):
                              new_tokens[-1] = new_tokens[-1] + token[2:]
                          else:
                              new_labels.append(tag_values[label_idx])
                              new_tokens.append(token)
################################################################################
##         post processing of the tokens to extract the relavant mention      ##
################################################################################
                      elem, span_token_label,countforid  = Postprocssing.processtokens(new_tokens,new_labels,test_sentence,elem,trigger_si,countforid)

            docs.append(span_token_label)
    
    tree.write(filepath1)
#Postprocssing.writetofile(docs, "../data/svm/trained_tagged.csv")
################################################################################
##         write the tagged mentions to xml                                   ##
################################################################################
allfiles = []
new_file = []
for file in os.listdir(path_pretty):
    filepath = os.path.join(path_pretty, file)
    with open(filepath, mode="r", encoding="utf-8") as fp:
        contents = fp.read()
        soup = BeautifulSoup(contents, 'xml')
        allfiles.append(soup)
        new_file.append(file)
################################################################################
##         Pretty print the xmls for redability                               ##
################################################################################
#print(len(allfiles))
for testfilesidx in range(0, len(allfiles)):
    filepath1 =os.path.join(path_guess , new_file[testfilesidx])
    testfile = open(filepath1, "w", encoding="utf-8")
    testfile.write(allfiles[testfilesidx].prettify())
##########################
Postprocssing.writetofile(docs, "../data/eval/trainin_csv/trained_tagged.csv")
