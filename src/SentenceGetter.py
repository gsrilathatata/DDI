##############################################################################
##   This class consolidats the sentences based on sentence numbers         ##
##############################################################################
## Sample code is picked from this website and some customization is done according to the need.
##https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/
class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, t) for w, t in zip(s["token"].values.tolist(),
                                                           s["tag"].values.tolist())]
        self.grouped = self.data.groupby("sentence_no").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
