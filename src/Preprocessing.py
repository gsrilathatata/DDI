import os
#sample code from https://github.com/seanysull/Drug-NER-and-Interaction-Extraction/blob/master/drugNER.py
import nltk
from nltk.tokenize import TreebankWordTokenizer
from bs4 import BeautifulSoup
from bs4.element import Tag
import Util
def getFollowLabel(label):
    In_labels = {'B-Precipitant': 'I-Precipitant',
                 'B-SpecificInteraction': 'I-SpecificInteraction',
                 'B-Trigger': 'I-Trigger',
                 'DB-Trigger': 'DI-Trigger',
                 'DB-Precipitant': 'DI-Precipitant'
                 }
    return In_labels[label]
#######################################################################################
def getTokensAndOffsets(sentence):
    """
    Given a sentence, as a string tokenise it and return tokens and token spans as a list of tuples
    """
    tokens = TreebankWordTokenizer().tokenize(sentence)
    offsets = TreebankWordTokenizer().span_tokenize(sentence)
    tokens_and_offsets = list(zip(tokens, offsets))
    return tokens_and_offsets

def multispantag(offset_label_dict,doublespanarrays):
    # get list of keys(offsets) from offset-drug_type dict these are the locations of the entities
    keys = list(sorted(offset_label_dict))
    ovelappingspan = []
    thislist = []
    disjoint_dict = {}
    for spans in doublespanarrays:
        spanlist = spans[0]
        biolabel = dtype_labels[spans[1]]
        for span in spanlist:
           if len(keys) != 0:
               k1 = keys[0][0]
               k2 = keys[0][1]
               if k1 <= span[0] and span[0] +span[1] <= k2:
                  thislist = spanlist
        for sss in thislist:
            ovelappingspan.append(int(sss[0]))
            ovelappingspan.append(int(sss[0]) + int(sss[1]))
        if len(ovelappingspan) > 0 :
            offset_full = (min(ovelappingspan),max(ovelappingspan))
            #remove the existing continous key and add the complete overlaping string
            if(k1,k2) in offset_entity_dict.keys():
                offset_entity_dict.pop((k1,k2))
                offset_entity_dict[offset_full] = biolabel
        if len(ovelappingspan) == 0:
            begin_disjoint = 0
            for span in spanlist:
                key = (span[0],span[0]+span[1])
                if begin_disjoint == 0:
                   disjoint_dict[key] = 'DB-' + spans[1]
                   begin_disjoint = 1
                elif begin_disjoint == 1:
                    disjoint_dict[key] = 'DI-' + spans[1]
    return offset_entity_dict, disjoint_dict

def assignLabel(index_token_offset, offset_label_dict,text_id,disjointlist):

    # creat empty list ready to append word plus label
    token_plus_biolabel = []
    # get list of keys(offsets) from offset-drug_type dict these are the locations of the entities
    keys = list(sorted(offset_label_dict))
    keys_disjoint = list(sorted(disjointlist))

    # flag to say we are in a sequence ie a multi word drug name
    in_sequence_flag = 0

    for (index, token, offset) in index_token_offset:
        # we delete the entity indexes(keys) as we go so we check if we have deleted all keys
        # if so we need to label all remaining terms as 'O' -  other
        if len(keys) != 0:
            k1 = keys[0][0]
            k2 = keys[0][1]

        elif len(keys) == 0:
            token_plus_biolabel.append((token,text_id,offset[0],'O'))
            continue

        if offset in  keys_disjoint:
            for key in keys_disjoint:

               if offset[0] == key[0] and offset[1] == key[1]:
                   label_dis = disjointlist[key]
                   token_plus_biolabel.append((token, text_id, offset[0],label_dis))
                   del keys_disjoint[0]
                   print("disjoint")
            continue

        if in_sequence_flag == 0:
            # if current start of token offset is less than that of current key(entity)
            # then assign other label and continue
            if k1 > offset[0]:
                token_plus_biolabel.append((token,text_id,offset[0],'O'))

            #elif k1 == offset[0] and k2 == offset[1] - 1:
            elif k1 == offset[0] and k2 == offset[1]:
                # get label then append token and label to list
                label = offset_label_dict[(k1, k2)]
                token_plus_biolabel.append((token,text_id, k1,label))
                # delete the matching key as it is no longer needed
                del keys[0]

            #elif k1 == offset[0] and k2 > offset[1] - 1:
            elif k1 == offset[0] and k2 > offset[1]:
                # get label for first token in multi worder
                label1 = offset_label_dict[(k1, k2)]
                token_plus_biolabel.append((token,text_id,k1, label1))
                in_sequence_flag = 1

        else:
            # check if word is the terminal of a sequence:
            # if so delete key and change flag  otherwise just label and continue
            if k2 == offset[1]:
                label_init = offset_label_dict[(k1, k2)]
                label_follow = getFollowLabel(label_init)
                token_plus_biolabel.append((token,text_id, k1,label_follow))
                in_sequence_flag = 0
                del keys[0]
            # token is in middle of sequence so just add follow label and proceed to next word
            else:
                label_init = offset_label_dict[(k1, k2)]
                label_follow = getFollowLabel(label_init)
                token_plus_biolabel.append((token,text_id,k1, label_follow))

    return token_plus_biolabel

path = "./training_data"

count = 0
allfiles = []
for file in os.listdir(path):
    filepath = os.path.join(path, file)
    with open(filepath, mode="r", encoding="utf-8") as fp:
        contents = fp.read()
        soup = BeautifulSoup(contents, 'html.parser')
        allfiles.append(soup)

docs = []
dtype_labels = {'Trigger': 'B-Trigger',
                'Precipitant': 'B-Precipitant',
                'SpecificInteraction': 'B-SpecificInteraction',
                'DTrigger': 'DB-Trigger',
                'DPrecipitant': 'DB-Precipitant',
                'other': 'O'}

for sss in range(0, len(allfiles)):
    ss = allfiles[sss].find_all("sentence")
    for elem in ss:
        text_id = elem.attrs["id"]
        entities = []
        doublespanarrays = []
        for child in elem:
            if type(child) == Tag:
                if child.name == "sentencetext":
                    text = (child.text).strip()
                    tokens_and_offsets = getTokensAndOffsets(text)
                    just_tokens = [token for (token, offset) in tokens_and_offsets]
                    just_offsets = [offset for (token, offset) in tokens_and_offsets]
                    indexes = list(range(len(just_tokens)))
                    tokens_plus_POS = nltk.pos_tag(just_tokens)
                    #tags = [t for (_, t) in tokens_plus_POS]
                    index_token_offset = list(zip(indexes, just_tokens, just_offsets))
                if child.name == "mention":
                    label = child.attrs['type']
                    if label == 'Precipitant' or label == 'Trigger':
                      entities.append(child)
        entity_dictionaries = [entity.attrs for entity in entities]
        if entities == []:
          offset_entity_dict = {}
        # if ntities is not empty proceed to add tags for words as appropriate
        elif entities != []:
            offset_entity_dict = {}
            for entity in entities:
                # build dictionary of offset:entity type for all entities in sentence
                offsetstring = entity.attrs['span']
                spans = [tuple(map(int, span.split()[:2])) for span in
                         offsetstring.split(';')]  # type: List[(int, int)]
                # split offset string on the hyphen, if length is > 3 then ignore as it is difficult
                # and rare edge case
                if len(spans) > 1:
                    doublespanarrays.append((spans,entity.attrs['type']))
                offset = offsetstring.split(" ")
                if len(offset) > 2:
                    continue
                # convert to tuple of integers
                offset_start = int(offset[0])
                offset_end = int(offset[1])
                offset_full = (offset_start, offset_start+offset_end)
                drugtype = entity.attrs['type']
                biolabel = dtype_labels[drugtype]
                offset_entity_dict[offset_full] = biolabel
        new_offset_entity_dict, disjointlist = multispantag( offset_entity_dict, doublespanarrays)
        token_plus_biolabel = assignLabel(index_token_offset, new_offset_entity_dict,text_id,disjointlist)

        docs.append(token_plus_biolabel)

Util.writetofile(docs,"svm/train.csv")
#Util.writetofile(docs,"svm/train_si.csv")