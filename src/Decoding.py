import spacy
import en_core_web_sm
## sample code is taken from spacy web examples
## this class extracts the overlapping mentions using spacy dependency parser
def getOverlapMentions(about_interest_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(about_interest_text)
    conjunct_array = []
    root_count = 0
    root_sent = ''
    #print('indise',about_interest_text)
    for token_len in range(0,len(doc)):
        if doc[token_len].dep_ == 'ROOT':
            root_sent = doc[token_len].text
            root_count = token_len
            conjunct_array.append(list(doc[token_len].conjuncts))

    left_edge = doc[root_count].left_edge
    segment1 = str(left_edge) + str(root_sent)

    segment2 =''
    for token in conjunct_array[0][0].lefts:
        segment2 = segment2 + token.text

    mention_text = []
    for count in range(len(conjunct_array[0])):
        mention_text.append((segment1 + ' ' + str(conjunct_array[0][count])))
        mention_text.append((segment2 + ' ' + str(conjunct_array[0][count])))

    return mention_text
