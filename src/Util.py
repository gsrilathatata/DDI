import csv
import os
from transformers import BertTokenizer

def writetofile(data,filename):
   if not os.path.exists(filename.split("/")[0]):
        os.makedirs(filename.split("/")[0])
   with open(filename,"w",encoding="utf-8",newline='') as csv_file:
     writer = csv.writer(csv_file, delimiter=',')
     writer.writerow(["token","sentence_no","start","tag"])
     for count in range(0,len(data)):
         for count1 in range(0,len(data[count])):
            writer.writerow(data[count][count1])

def writetoouputfile(data,filename):
   if not os.path.exists(filename.split("/")[0]):
        os.makedirs(filename.split("/")[0])
   with open(filename,"w") as csv_file:
     writer = csv.writer(csv_file, lineterminator='\n')
     for val in data:
       writer.writerow([val])

def writetoouputfile1(data,filename):
   if not os.path.exists(filename.split("/")[0]):
        os.makedirs(filename.split("/")[0])
   with open(filename,"w") as csv_file:
     writer = csv.writer(csv_file, lineterminator='\n')
     for val in data:
       writer.writerow([val])

def fileexists(filename):
    return os.path.exists(filename)

def tokenize_and_preserve_labels(sentence, text_labels,tokenizer):
    tokenized_sentence = []
    labels = []
    for word, label in zip(sentence, text_labels):

        # Tokenize the word and count # of subwords the word is broken into
        tokenized_word = tokenizer.tokenize(word)
        n_subwords = len(tokenized_word)

        # Add the tokenized word to the final tokenized word list
        tokenized_sentence.extend(tokenized_word)

        # Add the same label to the new list of labels `n_subwords` times
        labels.extend([label] * n_subwords)

    return tokenized_sentence, labels
def processtokens(new_tokens,new_labels,test_sentence,elem,soup,type_of_mention):
    span_token_label = []
    countforid = 0
    for (t, l) in zip(new_tokens, new_labels):
        if t != '[CLS]' and t != '[SEP]':
            start = test_sentence.find(t)
            end = len(t)
            span_token_label.append(([start, end], t, l))

    mentiontext = []
    for count in range(0, len(span_token_label) - 1):
        if type_of_mention != 'SpecificInteractions':
            if span_token_label[count][2] == 'B-Precipitant':
                if span_token_label[count + 1][2] == 'I-Precipitant':
                    span_token_label[count + 1] = list(span_token_label[count + 1])
                    span_token_label[count] = list(span_token_label[count])
                    span_token_label[count + 1][0] = list(span_token_label[count + 1][0])
                    span_token_label[count + 1][0][0] = span_token_label[count][0][0]
                    span_token_label[count + 1][0][1] = span_token_label[count][0][1] + span_token_label[count + 1][0][1]
                    span_token_label[count + 1][1] = span_token_label[count][1] + ' ' + span_token_label[count + 1][1]
                    span_token_label[count][1] = ' '
                    span_token_label[count][2] = 'XXXX'
                    span_token_label[count + 1][2] = 'B-Precipitant'
                else:
                    mentiontext.append(span_token_label[count])

            if span_token_label[count][2] == 'B-Trigger':
                if span_token_label[count + 1][2] == 'I-Trigger':
                    span_token_label[count + 1] = list(span_token_label[count + 1])
                    span_token_label[count] = list(span_token_label[count])
                    span_token_label[count + 1][0] = list(span_token_label[count + 1][0])
                    span_token_label[count + 1][0][0] = span_token_label[count][0][0]
                    span_token_label[count + 1][0][1] = span_token_label[count][0][1] + span_token_label[count + 1][0][1]
                    span_token_label[count + 1][1] = span_token_label[count][1] + ' ' + span_token_label[count + 1][1]
                    span_token_label[count][1] = ' '
                    span_token_label[count][2] = 'XXXX'
                    span_token_label[count + 1][2] = 'B-Trigger'
                else:
                    mentiontext.append(span_token_label[count])

            if (span_token_label[-1][2] == 'B-Precipitant'):
                mentiontext.append(span_token_label[-1][1])

            if (span_token_label[-1][2] == 'B-Trigger'):
                mentiontext.append(span_token_label[-1][1])
        elif type_of_mention == 'SpecificInteractions':
            if span_token_label[count][2] == 'B-SpecificInteraction':
                if span_token_label[count + 1][2] == 'I-SpecificInteraction':
                    span_token_label[count + 1] = list(span_token_label[count + 1])
                    span_token_label[count] = list(span_token_label[count])
                    span_token_label[count + 1][0] = list(span_token_label[count + 1][0])
                    span_token_label[count + 1][0][0] = span_token_label[count][0][0]
                    span_token_label[count + 1][0][1] = span_token_label[count][0][1] + span_token_label[count + 1][0][1]
                    span_token_label[count + 1][1] = span_token_label[count][1] + ' ' + span_token_label[count + 1][1]
                    span_token_label[count][1] = ' '
                    span_token_label[count][2] = 'XXXX'
                    span_token_label[count + 1][2] = 'B-SpecificInteraction'
                else:
                    mentiontext.append(span_token_label[count])

            if (span_token_label[-1][2] == 'B-SpecificInteraction'):
                mentiontext.append(span_token_label[-1][1])

    if len(mentiontext) > 0:
        for mention in mentiontext:
            new_tag = soup.new_tag("Mention", "new tag added")
            countforid = countforid + 1
            new_tag.attrs["id"] = "M1" + str(countforid)
            new_tag.attrs["type"] = mention[2][2:]
            new_tag.attrs["span"] = str(mention[0][0]) + ' ' + str(mention[0][1])
            new_tag.attrs["str"] = mention[1]
            elem.append(new_tag)
            elem.append("\n")
    return elem