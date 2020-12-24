import csv
import os
from transformers import BertTokenizer
import Decoding
from xml.etree.ElementTree import ElementTree, SubElement
################################################################################
## The main class for post processing the mentions outputafter predictions    ##
################################################################################

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


## this function removes the mention from GOLD dataset and copies the files generated
## to TEST folder. The training algorithm pick up from this folder for annotating and
## annotated files are copied to GUESS folder
##evanlution python script compares both the files to provide the details
def removeMention(gold,guess):
    #path = "GOLD"
    #path1 = "TEST"
    for file in os.listdir(gold):
        filepath = os.path.join(gold, file)
        filepath1 = os.path.join(guess, file)
        tree = ElementTree()
        tree.parse(filepath)

        sentences = tree.findall('.//Sentence')
        for child in sentences:
            mentions = child.findall('Mention')
            for mention in mentions:
                child.remove(mention)

        tree.write(filepath1)

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
def processtokens(new_tokens,new_labels,test_sentence,elem,type_of_mention,countforid):
    span_token_label = []
    span_token_label_test = []
    text_id = elem.attrib["id"]
    #countforid = 0
    for (t, l) in zip(new_tokens, new_labels):
        if t != '[CLS]' and t != '[SEP]':
            start = test_sentence.find(t)
            end = len(t)
            span_token_label.append(([start, end], t, l,text_id))
            span_token_label_test.append((t,text_id, start, l))
           # span_token_label_test.append(([start, end], t, l,text_id))

    #`print( span_token_label)
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
    for i in range(0, len(span_token_label)):
        if span_token_label[i][2] == 'DB-Trigger':
            span_token_label[i] = list(span_token_label[i])

            for j in range(i + 1, len(span_token_label)):
                if span_token_label[j][2] == 'DB-Trigger' or j == len(span_token_label) - 1:
                    span_token_label[j] = list(span_token_label[j])
                    span_token_label[i][0] = str(span_token_label[i][0]).replace(' [ ', '').replace(']', '').replace(',', '')
                    mentiontext.append(span_token_label[i])
                    #print(mentiontext)
                    break
                elif span_token_label[j][2] == 'DI-Trigger':
                    span_token_label[j] = list(span_token_label[j])
                    span_token_label[i][2] = ' '
                    span_token_label[j][2] = 'DB-Trigger'
                    span_token_label[j][1] = span_token_label[i][1] + ' | ' + span_token_label[j][1]
                    span_token_label[j][0] = str(span_token_label[i][0]).replace('[', '').replace(']', '').replace(',', '') + ';' + str(
                        span_token_label[j][0]).replace('[', '').replace(']', '').replace(',', '')
                    span_token_label[i][1] = ' '
                    break

    if len(mentiontext) > 0:
        #print(mentiontext)
        for mention in range(0, len(mentiontext)-1):
          new_tag =  SubElement(elem, "Mention")
          countforid = countforid + 1
          #print(str(mentiontext[mention][0]))
          new_tag.attrib["id"] = "M" + str(countforid)
          try:
              typofmention = mentiontext[mention][2][2:].replace("-Trigger","Trigger")
              new_tag.attrib["type"] = typofmention#mentiontext[mention][2][2:].replace("-Trigger","Trigger")
              new_tag.attrib["span"] = str(mentiontext[mention][0]).replace('[', '').replace(']', '').replace(',', '')#str(mentiontext[mention][0][0]) + ' ' + str(mentiontext[mention][0][1])
              new_tag.attrib["str"] = mentiontext[mention][1]
              if typofmention == 'Precipitant':
                 # iday this cod coms from som drug rpository(this is not inetgrated to handle to get id hence using the same sequence mention id 
                 new_tag.attrib["code"] = "N" +  str(countforid)
              splittext = mentiontext[mention][1].split()
              #print(mentiontext)
              if len(splittext) > 3:
                 #print(splittext)
                 multiplementions =  Decoding.getOverlapMentions(mentiontext[mention][1]) #in this case i need to handle the depenecy parser situation
                 # the mutiple mentions needs to be added as separate mentions
                 # this is pending task
          except Exception as e:
              pass 
          #elem.append(new_tag)
         # elem.append("\n")
    return elem ,  span_token_label_test,countforid
