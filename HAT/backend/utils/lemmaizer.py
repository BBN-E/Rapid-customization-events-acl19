import re

class Lemmaizer(object):
    def __init__(self,lemma_nv_path):
        self.lemma_table = dict()
        with open(lemma_nv_path,'r') as fp:
            for i in fp:
                i = i.strip()
                transformed, original = i.split(" ")
                self.lemma_table[transformed] = original

    def get_lemma(self,string,multi_word=False):
        string = string.strip()
        if multi_word is True:
            pattern = re.compile(r'[\t\n\r.,]+')
            string = pattern.sub("",string)
            string = string.replace(" ","_")
            string = string.lower()
            return string
        else:
            pattern = re.compile(r'[\t\n\r.,_ ]+')
            string = pattern.sub("", string)
            string = string.lower()
            return  self.lemma_table.get(string,string)

# lemmaizer = Lemmaizer("/home/hqiu/massive/CauseEx-novel-event-ui/java/causeex/causeex-common/src/main/resources/com/bbn/causeex/common/lemma.nv")