import spacy
from spacy import displacy
about_interest_text = ('Combined P-gp and strong CYP3A inhibitors and other drugs that, like XARELTO ,impair hemostasis increases the risk of bleeding.')
nlp = spacy.load('en_core_web_sm')
about_interest_doc = nlp(about_interest_text)
#displacy.serve(about_interest_doc, style='dep')
for token in about_interest_doc:
    print(token.text, token.tag_, token.head.text, token.dep_)