

from collections import defaultdict

from nlplingo.common.utils import JsonObject
from nlplingo.common.utils import Struct


pos_types = Struct(NOUN='NOUN', VERB='VERB', ADJ='ADJ', ADV='ADV', MISC='MISC')


def interpret_pos_type(p):
    if p == 'n':
        return pos_types.NOUN
    elif p == 'v':
        return pos_types.VERB
    elif p == 'a':
        return pos_types.ADJ
    elif p == 'r':
        return pos_types.ADV
    else:
        return None


wordnet_types = Struct(Antonym='Antonym', Hypernym='Hypernym', InstanceHypernym='InstanceHypernym', Hyponym='Hyponym',
                       InstanceHyponym='InstanceHyponym', MemberHolonym='MemberHolonym',
                       SubstanceHolonym='SubstanceHolonym', PartHolonym='PartHolonym', MemberMeronym='MemberMeronym',
                       SubstanceMeronym='SubstanceMeronym', PartMeronym='PartMeronym', Attribute='Attribute',
                       DerivationalForm='DerivationalForm', DomainSynsetTopic='DomainSynsetTopic',
                       MemberDomainTopic='MemberDomainTopic', DomainSynsetRegion='DomainSynsetRegion',
                       MemberDomainRegion='MemberDomainRegion', DomainSynsetUsage='DomainSynsetUsage',
                       MemberDomainUsage='MemberDomainUsage', Entailment='Entailment', Cause='Cause', AlsoSee='AlsoSee',
                       VerbGroup='VerbGroup')


class Pointer(object):
    def __init__(self, type, synset_offset, pos_type):
        """
        :type type: str
        :type synset_offset: str
        :type pos_type: str
        """
        self.type = type
        self.synset_offset = synset_offset
        self.pos_type = pos_type

    def to_string(self):
        return '({} {} {})'.format(self.type, self.synset_offset, self.pos_type)


class Synset(object):
    def __init__(self, offset):
        """
        :type offset: str
        """
        self.offset = offset
        self.words = []
        """:type: list[str]"""
        self.pointers = []
        """:type: list[wordnet.Pointer]"""
        self.lex_file_num = None
        """:type: str"""
        self.pos_type = None
        """:type: str"""
        self.gloss = None
        """:type: str"""

    def add_word(self, w):
        self.words.append(w)

    def add_pointer(self, p):
        self.pointers.append(p)

    def __hash__(self):
        return hash((self.offset))

    def __eq__(self, other):
        return (self.offset) == (other.offset)

    def synonyms_and_gloss_string(self):
        return '{} -- ({})'.format(', '.join(self.words), self.gloss)

    def get_hypernym_pointers(self):
        ret = []
        for ptr in self.pointers:
            if WordNetManager.interpret_pointer_symbol(ptr.type) == wordnet_types.Hypernym:
                ret.append(ptr)
        return ret

    def get_hyponym_pointers(self):
        ret = []
        for ptr in self.pointers:
            if WordNetManager.interpret_pointer_symbol(ptr.type) == wordnet_types.Hyponym:
                ret.append(ptr)
        return ret

    def to_string(self):
        return '{} {} {} pointers:{}'.format(self.offset, self.pos_type, ' '.join(self.words), ','.join(p.to_string() for p in self.pointers))


class SynsetInfo(JsonObject):
    def __init__(self, lemma, lexsn, pos_category, gloss):
        self.lemma = lemma
        self.lexsn = lexsn
        self.pos_category = pos_category
        self.gloss = gloss
        self.synonyms = []
        """:type: list(str)"""
        self.siblings = []
        """:type: list[SynsetWords]"""
        self.hypernym_paths = []
        """:type: list[list[SynsetWords]]"""
        self.hyponym_paths = []
        """:type: list[list[SynsetWords]]"""

    def add_synonym(self, w):
        """
        :type w: str
        """
        self.synonyms.add(w)

    def add_sibling(self, sibling):
        """
        :type sibling: SynsetWords
        """
        self.siblings.append(sibling)

    def add_hypernym_path(self, path):
        """
        :type path: list[SynsetWords]
        """
        self.hypernym_paths.append(path)

    def add_hyponym_path(self, path):
        """
        :type path: list[SynsetWords]
        """
        self.hyponym_paths.append(path)


class Sense(object):
    def __init__(self, id, synset):
        """
        :type id: str
        :type synset: Synset
        """
        self.id = id
        self.synset = synset


class Word(object):
    def __init__(self, number_of_senses):
        self.senses = [None] * number_of_senses
        """:type: list[Sense]"""
        self.pos_type = None
        """:type: str"""
        self.lemma = None
        """:type: str"""

    def add_sense(self, sense, i):
        self.senses[i-1] = sense

    def get_sense(self, i):
        return self.senses[i-1]

    def number_of_senses(self):
        return len(self.senses)

    def get_first_sense_synset_id(self):
        sense1 = self.senses[0]
        if sense1 is not None:
            return sense1.synset.offset
        else:
            return None

    def synonyms_and_gloss_string(self):
        ret = []

        if self.pos_type == pos_types.NOUN:
            ret.append('The noun {} has {} senses'.format(self.lemma, len(self.senses)))
        elif self.pos_type == pos_types.VERB:
            ret.append('The verb {} has {} senses'.format(self.lemma, len(self.senses)))

        for i, sense in enumerate(self.senses):
            if sense.synset is not None:
                ret.append('{}. {}'.format((i+1), sense.synset.synonyms_and_gloss_string()))
            else:
                ret.append('{}. NULL synset'.format(i+1))

    def first_sense_synonyms_and_gloss_string(self):
        return '1. {}'.format(self.senses[0].synset.synonyms_and_gloss_string())


class WordNetManager(object):
    def __init__(self, noun_datafile, verb_datafile, sense_indexfile):
        self.noun_synsets = dict()
        """:type: Dict[str, Synset]"""
        self.verb_synsets = dict()
        """:type: Dict[str, Synset]"""

        noun_syn = self.read_data_file(noun_datafile)
        for syn in noun_syn:
            self.noun_synsets[syn.offset] = syn

        verb_syn = self.read_data_file(verb_datafile)
        for syn in verb_syn:
            self.verb_synsets[syn.offset] = syn

        nouns, verbs = self.read_sense_index_file(sense_indexfile)
        self.noun_words = self.populate_words(nouns, pos_types.NOUN)
        """:type: dict[str, Word]"""
        self.verb_words = self.populate_words(verbs, pos_types.VERB)
        """:type: dict[str, Word]"""

    class SenseInfo(object):
        def __init__(self, sense_num, sense_id, synset_offset):
            self.sense_num = sense_num
            self.sense_id = sense_id
            self.synset_offset = synset_offset

    @classmethod
    def interpret_pointer_symbol(cls, ptr):
        if ptr == '!':
            return wordnet_types.Antonym
        elif ptr == '@':
            return wordnet_types.Hypernym
        elif ptr == '@i':
            return wordnet_types.InstanceHypernym
        elif ptr == '~':
            return wordnet_types.Hyponym
        elif ptr == '~i':
            return wordnet_types.InstanceHyponym
        elif ptr == '#m':
            return wordnet_types.MemberHolonym
        elif ptr == '#s':
            return wordnet_types.SubstanceHolonym
        elif ptr == '#p':
            return wordnet_types.PartHolonym
        elif ptr == '%m':
            return wordnet_types.MemberMeronym
        elif ptr == '%s':
            return wordnet_types.SubstanceMeronym
        elif ptr == '%p':
            return wordnet_types.PartMeronym
        elif ptr == '=':
            return wordnet_types.Attribute
        elif ptr == '+':
            return wordnet_types.DerivationalForm
        elif ptr == ';c':
            return wordnet_types.DomainSynsetTopic
        elif ptr == '-c':
            return wordnet_types.MemberDomainTopic
        elif ptr == ';r':
            return wordnet_types.DomainSynsetRegion
        elif ptr == '-r':
            return wordnet_types.MemberDomainRegion
        elif ptr == ';u':
            return wordnet_types.DomainSynsetUsage
        elif ptr == '-u':
            return wordnet_types.MemberDomainUsage
        elif ptr == '*':
            return wordnet_types.Entailment
        elif ptr == '>':
            return wordnet_types.Cause
        elif ptr == '^':
            return wordnet_types.AlsoSee
        elif ptr == '$':
            return wordnet_types.VerbGroup
        else:
            return None

    def get_senses_for_lemma(self, lemma, pos_type):
        """
        :type lemma: str
        :type pos_str: str
        Returns: list[Sense]

        pos_type: this is just 'n', 'v', etc.
        """
        if pos_type == pos_types.NOUN and lemma in self.noun_words:
            return self.noun_words[lemma].senses
        elif pos_type == pos_types.VERB and lemma in self.verb_words:
            return self.verb_words[lemma].senses
        else:
            return None

    def read_data_file(self, filepath):
        synsets = []
        """:type: list[Synset]"""

        with open(filepath, 'r') as f:
            lines = [line.rstrip() for line in f]

        # skip comment lines at the top of the file
        line_index = 0
        while lines[line_index].startswith('  '):
            line_index += 1

        for line in lines[line_index:]:
            i1 = line.index('|')
            gloss = line[i1+1:]
            tokens = line[0:i1].split()

            synset = Synset(tokens[0])
            synset.gloss = gloss
            synset.lex_file_num = tokens[1]
            synset.pos_type = interpret_pos_type(tokens[2])

            word_count = int(tokens[3], 16)
            i = 4
            for j in range(word_count):
                synset.add_word(tokens[i])
                i += 2

            pointer_count = int(tokens[i])
            for j in range(pointer_count):
                pointer_symbol = tokens[i+1]
                synset_offset = tokens[i+2]
                pos = tokens[i+3]
                source_target = tokens[i+4]
                ptr = Pointer(pointer_symbol, synset_offset, interpret_pos_type(pos))
                synset.add_pointer(ptr)
                i += 4

            synsets.append(synset)
        return synsets

    def read_sense_index_file(self, filepath):
        nouns = defaultdict(list)
        """:type: Dict[str, list[SenseInfo]]"""
        verbs = defaultdict(list)
        """:type: Dict[str, list[SenseInfo]]"""

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f]

        for line in lines:
            tokens = line.split()

            sense_id = tokens[0]
            lemma = sense_id[0: sense_id.index('%')]
            synset_offset = tokens[1]
            sense_num = int(tokens[2])

            if '%1' in sense_id:
                nouns[lemma].append(self.SenseInfo(sense_num, sense_id, synset_offset))
            elif '%2' in sense_id:
                verbs[lemma].append(self.SenseInfo(sense_num, sense_id, synset_offset))

        return (nouns, verbs)

    def populate_words(self, source, pos_type):
        """
        :type source: Dict[str, list[SenseInfo]]
        """
        ret = dict()
        """:type: Dict[str, Word]"""

        for lemma in source.keys():
            sense_info_list = source[lemma]

            w = Word(len(sense_info_list))
            w.lemma = lemma
            w.pos_type = pos_type

            for sense_info in sense_info_list:
                sense_num = sense_info.sense_num
                sense_id = sense_info.sense_id
                synset_offset = sense_info.synset_offset

                synset = None
                if pos_type == pos_types.NOUN:
                    synset = self.noun_synsets[synset_offset]
                elif pos_type == pos_types.VERB:
                    synset = self.verb_synsets[synset_offset]
                else:
                    raise ValueError('pos_type should be either NOUN or VERB, but it is {}'.format(pos_type))

                sense = Sense(sense_id, synset)
                w.add_sense(sense, sense_num)

            ret[lemma] = w
        return ret

    def get_sibling_synsets(self, synset):
        """Given a synset, get its sibling synsets, i.e. other children of its parent
        :type synset: Synset
        Returns: set(Synset)
        """
        ret = set()
        parent_synsets = self.get_hypernym_synsets(synset)
        for p in parent_synsets:
            for c in self.get_hyponym_synsets(p):
                # do not add yourself as your own sibling
                if c.offset != synset.offset:
                    ret.add(c)
        return ret

    def get_synset_for_pointer(self, ptr):
        synset = None
        if ptr.pos_type == pos_types.NOUN:
            synset = self.noun_synsets[ptr.synset_offset]
        elif ptr.pos_type == pos_types.VERB:
            synset = self.verb_synsets[ptr.synset_offset]
        return synset

    def get_hyponym_synsets(self, synset):
        ret = []
        for ptr in synset.get_hyponym_pointers():
            s = self.get_synset_for_pointer(ptr)
            if s is not None:
                ret.append(s)
        return ret

    def get_hypernym_synsets(self, synset):
        ret = []
        for ptr in synset.get_hypernym_pointers():
            s = self.get_synset_for_pointer(ptr)
            if s is not None:
                ret.append(s)
        return ret

    def get_all_hyponym_paths(self, synset, current_path, all_paths):
        """Get all hyponyms (children) of a synset
        :type synset: Synset
        :type current_path: list[Synset]
        :type all_paths: list[list[Synset]]
        """
        #print('get_all_hyponym_paths, synet={}'.format(synset.to_string()))
        children_synsets = self.get_hyponym_synsets(synset)
        if len(children_synsets) == 0:
            all_paths.append(current_path)
            return

        for child_synset in children_synsets:
            #print('P={}\tC={}'.format(synset.to_string(), child_synset.to_string()))
            synsets_on_current_path = set(s.offset for s in current_path)
            if child_synset.offset in synsets_on_current_path:
                all_paths.append(current_path)
            else:
                self.get_all_hyponym_paths(child_synset, current_path + [child_synset], all_paths)

        return

    def get_all_hypernym_paths(self, synset, current_path, all_paths):
        """Get all hypernyms (parents) of a synset
        :type synset: Synset
        :type current_path: list[Synset]
        :type all_paths: list[list[Synset]]
        """
        parent_synsets = self.get_hypernym_synsets(synset)
        if len(parent_synsets) == 0:
            all_paths.append(current_path)
            return

        for parent_synset in parent_synsets:
            synsets_on_current_path = set(s.offset for s in current_path)
            if parent_synset.offset in synsets_on_current_path:
                all_paths.append(current_path)
            else:
                self.get_all_hypernym_paths(parent_synset, current_path + [parent_synset], all_paths)

        return

    def get_synset_for_senseid(self, id):
        w = None
        """:type: wordnet.Word"""

        if '%1' in id:
            lemma = id[0: id.index('%1')]
            if lemma in self.noun_words:
                w = self.noun_words[lemma]
            else:
                print('Cannot find {} in noun_words'.format(lemma))
        elif '%2' in id:
            lemma = id[0: id.index('%2')]
            if lemma in self.verb_words:
                w = self.verb_words[lemma]
            else:
                print('Cannot find {} in verb_words'.format(lemma))
        else:
            raise ValueError('We only capture NOUN or VERB')

        if w is not None:
            for sense in w.senses:
                if sense.id == id:
                    return sense.synset

        return None


class SynsetWords(JsonObject):
    def __init__(self, words):
        self.words = []
        """:type: list(str)"""
        for w in words:
            self.words.append(w.replace('_', ' '))
















