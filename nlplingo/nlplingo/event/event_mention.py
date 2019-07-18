from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

from collections import defaultdict
import codecs
import re

import numpy as np

from nlplingo.event.event_argument import EventArgumentExample
from nlplingo.event.event_argument import EventArgumentGenerator
from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.common.io_utils import read_file_to_set
from nlplingo.text.dependency_relation import DependencyRelation
#from nlplingo.text.dependency_relation import find_shortest_dep_paths_between_tokens
#from nlplingo.annotation.stanford_corenlp import find_dep_paths_to_root
from nlplingo.annotation.stanford_corenlp import find_trigger_candidates
from nlplingo.nlp.srl_relation import find_srls_involving_span
from nlplingo.text.text_theory import SRL

class EventMentionGenerator(object):

    def __init__(self, event_domain, params, word_embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        self.word_embeddings = word_embeddings
        (none_text, none_idx, none_vec) = self.word_embeddings.get_none_vector()
        self.none_idx = none_idx
        #(none_text2, none_idx2, none_vec2) = self.word_embeddings.get_none_vector2()
        #self.none_idx2 = none_idx2

        self.event_domain = event_domain
        self.params = params
        self.max_sent_length = params.get_int('max_sent_length')
        self.neighbor_dist = params.get_int('cnn.neighbor_dist')
        # the following file gives the false negatives, so we should exclude them when generating negative examples
        if params.has_key('pair.false_negative'):
            self.false_negatives = self._read_false_negatives(params.get_string('pair.false_negative'))
        else:
            self.false_negatives = dict()
        """:type: dict(str, str)"""
        self.statistics = defaultdict(int)

        if params.has_key('trigger_words_toskip'):
            self.trigger_words_toskip = read_file_to_set(params.get_string('trigger_words_toskip'))
        else:
            self.trigger_words_toskip = set()

        if params.has_key('verb_words'):
            self.verb_words = read_file_to_set(params.get_string('verb_words'))
        else:
            self.verb_words = set()

        self.reg = re.compile('^[a-z\-]+$')


    def _read_false_negatives(self, filepath):
        ret = dict()
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                line = line[line.index(' ')+1:]     # skip the first token/column
                docid_sentence_num = line[0:line.index(' ')]
                sentence_text = line[line.index(' ')+1:]
                ret[docid_sentence_num] = sentence_text
        return ret

    def generate(self, docs):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        for doc in docs:
            for sent in doc.sentences:
                examples.extend(self._generate_sentence(sent))

        for k in sorted(self.statistics.keys()):
            v = self.statistics[k]
            print('EventMentionGenerator stats, {}:{}'.format(k, v))

        return examples

    def select_args(self, candidate_args):
        """
        :type candidate_args: list[nlplingo.text.text_span.EventArgument]
        """
        nnp_args = [arg for arg in candidate_args if arg.entity_mention.is_proper_noun()]
        nn_args = [arg for arg in candidate_args if arg.entity_mention.is_common_noun()]
        adj_args = [arg for arg in candidate_args if arg.entity_mention.is_adjective()]

        ret = []
        ret.extend(nnp_args)
        ret.extend(nn_args)
        ret.extend(adj_args)

        return [arg for arg in ret if (
            arg.entity_mention.label == 'PERSON' or arg.entity_mention.label == 'PERSON_DESC' or \
            arg.entity_mention.label == 'ORG' or arg.entity_mention.label == 'ORG_DESC' or \
            arg.entity_mention.label == 'ORGANIZATION' or arg.entity_mention.label == 'ORGANIZATION_DESC' or \
            arg.entity_mention.label == 'GPE' or arg.entity_mention.label == 'GPE_DESC')]

        #selected_arg = None
        # if len(selected_candidate_args) == 1:
        #     return selected_candidate_args
        # else:
        #     # if there are multiple args, we first, (i) filter for NNP, (ii) filter for those before the anchor, (iii) choose nearest the anchor
        #     nnp_args = []
        #     """:type: list[nlplingo.text.text_span.EventArgument]"""
        #     for arg in selected_candidate_args:
        #         postags = set()
        #         for token in arg.entity_mention.tokens:
        #             postags.add(token.spacy_token.tag_)
        #         if 'NNP' in postags:
        #             nnp_args.append(arg)
        #
        #     return nnp_args

        #     if len(nnp_args) == 1:
        #         return nnp_args[0]
        #     elif len(nnp_args) > 1:
        #         before_anchor_args = []
        #         for arg in nnp_args:
        #             if arg.start_char_offset() < anchor.start_char_offset():
        #                 before_anchor_args.append(arg)
        #         if len(before_anchor_args) == 0:  # choose the one nearest the anchor
        #             min_distance = sentence.end_char_offset() - sentence.start_char_offset()
        #             nearest_arg = None
        #             for arg in nnp_args:
        #                 distance = arg.start_char_offset() - anchor.start_char_offset()
        #                 if distance < min_distance:
        #                     min_distance = distance
        #                     nearest_arg = arg
        #             selected_arg = nearest_arg
        #         elif len(before_anchor_args) == 1:
        #             selected_arg = before_anchor_args[0]
        #         else:  # choose the one nearest the anchor
        #             min_distance = sentence.end_char_offset() - sentence.start_char_offset()
        #             nearest_arg = None
        #             for arg in before_anchor_args:
        #                 distance = anchor.start_char_offset() - arg.start_char_offset()
        #                 if distance < min_distance:
        #                     min_distance = distance
        #                     nearest_arg = arg
        #             selected_arg = nearest_arg
        # return selected_arg
    
    @classmethod
    def sentence_text_with_markup(cls, sentence, anchor, entity_mention):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        :type anchor: nlplingo.text.text_span.Anchor
        :type entity_mention: nlplingo.text.text_span.EntityMention
        """
        anchor_start = anchor.start_char_offset()
        anchor_end = anchor.end_char_offset()
        em_start = entity_mention.start_char_offset()
        em_end = entity_mention.end_char_offset()

        is_anchor = False
        ret = []
        ret.append(anchor.label)
        ret.append('[{}]'.format(anchor.head().text))

        """:type: list[str]"""
        for token in sentence.tokens:
            text = ''
            if token.start_char_offset() == anchor_start:
                text += 'T['
                is_anchor  = True
            elif token.start_char_offset() == em_start:
                text += 'A_{}['.format(entity_mention.label)

            text += token.text
            if is_anchor:
                text += '/{}'.format(token.pos_tag.encode('ascii', 'ignore'))

            if token.end_char_offset() == anchor_end or token.end_char_offset() == (anchor_end-1):
                text += ']'
                is_anchor = False
            elif token.end_char_offset() == em_end or token.end_char_offset() == (em_end-1):
                text += ']'
            ret.append(text)

        ret.append('|||')
        for r in anchor.head().dep_relations + anchor.head().child_dep_relations:
            #print(r.to_string())
            r_parts = r.to_string().split(':')
            r_name = r_parts[0:-2]
            r_parent_text = sentence.tokens[int(r_parts[-2])].text
            r_child_text = sentence.tokens[int(r_parts[-1])].text
            ret.append('{}:{}:{}'.format(r_name, r_parent_text, r_child_text))

        return ' '.join(ret).replace('\n', ' ').replace('\r', ' ')

    def _entity_mention_is_source(self, em, sentence):
        """
        :type em: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        """
        # TODO: use dependnency relations to decide whether this entity mention (em) is a Source argument of some event
        return True

    def _suggest_anchor_for_entity_mention(self, em, sentence):
        """
        :type em: nlplingo.text.text_span.EntityMention
        :type sentence: nlplingo.text.text_span.Sentence
        """
        # TODO: use dependency relations to suggest an anchor

        before_tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""
        after_tokens = []
        """:type: list[nlplingo.text.text_span.Token]"""
        for i, token in enumerate(sentence.tokens):
            if token.end_char_offset() < em.start_char_offset():
                before_tokens.append(token)
            elif token.start_char_offset() > em.end_char_offset():
                after_tokens.append(token)

        anchor_candidates = [token for token in after_tokens if token.pos_category() == 'VERB']
        if len(anchor_candidates) > 0:
            anchor = Anchor('dummy-id', IntPair(token.start_char_offset(), token.end_char_offset()), token.text, 'None')
            anchor.with_tokens([token])
            return anchor
        else:
            return None

    def _generate_sentence(self, sentence):
        """
        :type sentence: nlplingo.text.text_span.Sentence
        """
        ret = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() >= self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        for token in sentence.tokens:
            paths = DependencyRelation.find_dep_paths_to_root(token, sentence)
            for path in paths:
                token.add_dep_path_to_root(path)

        tokens = [None] + sentence.tokens

        if len(sentence.events)==0:
            #print('NEG: {}_{} {}'.format(sentence.docid, sentence.index, sentence.text))
            key = '{}_{}'.format(sentence.docid, sentence.index)
            if key not in self.false_negatives: # does not contain any of the event types we have annotated
                for em in sentence.entity_mentions:
                    # checking the em label to be GPE or ORGANIZATION (without _desc suffix), also ensures it is a proper noun
                    # we allow for adjective because some construction, e.g. 'Israeli planes', where 'Israeli' might be adjective
                    if (em.label == 'GPE' or em.label == 'ORGANIZATION') and (em.is_proper_noun() or em.is_common_noun() or em.is_adjective()):

                        srl_relations = find_srls_involving_span(em.head(), sentence, target_roles=set(['A0', 'A0:compound']))
                        """:type: list[nlplingo.text.text_theory.SRL]"""
                        srl_predicate_indices = [r.predicate_token.index_in_sentence for r in srl_relations if
                                          r.predicate_token != em.head()]

                        trigger_candidate_indices = find_trigger_candidates(em.head(), sentence,
                                                                     self.trigger_words_toskip, self.verb_words)

                        for trigger_index in set(trigger_candidate_indices.keys() + srl_predicate_indices):
                            token = sentence.tokens[trigger_index]
                            anchor = Anchor('dummy-id', IntPair(token.start_char_offset(), token.end_char_offset()),
                                            token.text, 'None')
                            anchor.with_tokens([token])

                            example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, 'None')
                            EventArgumentGenerator._generate_example(example, tokens, self.max_sent_length, self.neighbor_dist)
                            ret.append(example)
                            self.statistics['#None_Source'] += 1

        if len(sentence.events) > 0:
            print('SENT: {}'.format(sentence.to_string()))
        for event in sentence.events:
            anchor = event.anchors[0]

            #print('ANCHOR INFO: {} {} {} {}'.format(anchor.head().text, anchor.head().lemma, anchor.head().has_vector, anchor.head().vector_index))

            dep_string = ','.join(r.to_string() for r in anchor.head().dep_relations)
            #print('event_mention.py\tANCHOR\t{}\t{}\t{}'.format(event.label, anchor.text, dep_string))

            self.statistics['#{}'.format(event.label)] += 1
            candidate_args = [arg for arg in event.arguments if arg.label == 'Source']
            selected_args = self.select_args(candidate_args)

            print('ANCHOR: {}/{} ||| {}'.format(anchor.head().to_string(), anchor.head().pos_tag, anchor.to_string()))
            #print('SENT: {}'.format(sentence.to_string()))
            #for arg in candidate_args:
            #    print('Candidate ENTITY: {}/{}/{} ||| {}'.format(arg.entity_mention.head().to_string(),
            #                                           arg.entity_mention.head().pos_tag,
            #                                           arg.entity_mention.head_pos_category(),
            #                                           arg.entity_mention.to_string()))


            for selected_arg in selected_args:
                print('Selected ENTITY: {}/{}/{} ||| {}'.format(selected_arg.entity_mention.head().to_string(),
                                                                 selected_arg.entity_mention.head().pos_tag,
                                                                 selected_arg.entity_mention.head_pos_category(),
                                                                 selected_arg.entity_mention.to_string()))

                #print('ANCHOR: {} ||| {}'.format(anchor.head().to_string(), anchor.to_string()))
                #print('ENTITY: {} ||| {}'.format(selected_arg.entity_mention.head().to_string(), selected_arg.entity_mention.to_string()))
                #print('SENT: {}'.format(sentence.to_string()))

                srl_relations = find_srls_involving_span(selected_arg.entity_mention.head(), sentence, target_roles=set(['A0', 'A0:compound']))
                """:type: list[nlplingo.text.text_theory.SRL]"""
                srl_predicates = [r.predicate_token for r in srl_relations if r.predicate_token != selected_arg.entity_mention.head()]
                for predicate in srl_predicates:
                    if (predicate.pos_tag.startswith('VB') or predicate.pos_tag == 'NN') and (predicate.lemma not in self.trigger_words_toskip):
                        if not self.reg.match(predicate.lemma):
                            print(' - *SRL predicate {}:{}/{}/{}'.format(predicate.index_in_sentence, predicate.text, predicate.lemma, predicate.pos_tag))
                        else:
                            print(' - SRL predicate {}:{}/{}/{}'.format(predicate.index_in_sentence, predicate.text, predicate.lemma, predicate.pos_tag))


                dep_paths = DependencyRelation.find_shortest_dep_paths_between_tokens(
                    selected_arg.entity_mention.head(), anchor.head(), sentence.tokens)
                for dep_path in dep_paths:
                    simple_path = []
                    for r in dep_path:
                        parts = r.split(':')
                        simple_path.append('{}:{}'.format(parts[0], parts[1]))

                    ##print(' - COMPLEX-DEP-PATH\t{}\t{}'.format(selected_arg.entity_mention.head().to_string(),
                    ##                                           ' '.join(r for r in dep_path)))
                    ##print(' - SIMPLE-DEP-PATH\t{}\t{}'.format(selected_arg.entity_mention.head().to_string(),
                    ##                                   ' '.join(r for r in simple_path)))

                trigger_candidates = find_trigger_candidates(selected_arg.entity_mention.head(), sentence, self.trigger_words_toskip, self.verb_words)
                for trigger_candidate in trigger_candidates:
                    pos_tag = sentence.tokens[trigger_candidate].pos_tag
                    if (pos_tag.startswith('VB') or pos_tag == 'NN') and (sentence.tokens[trigger_candidate].lemma not in self.trigger_words_toskip):
                        if not self.reg.match(sentence.tokens[trigger_candidate].lemma):
                            print(' - *Trigger-candidate\t{}/{}'.format(sentence.tokens[trigger_candidate].text, sentence.tokens[trigger_candidate].pos_tag))
                        else:
                            print(' - Trigger-candidate\t{}/{}'.format(sentence.tokens[trigger_candidate].text, sentence.tokens[trigger_candidate].pos_tag))
                if len(trigger_candidates) == 0:
                    print(' - Trigger-candidate\tNone')

                example = EventArgumentExample(anchor, selected_arg.entity_mention, sentence, self.event_domain, self.params, selected_arg.label)
                EventArgumentGenerator._generate_example(example, tokens, self.max_sent_length, self.neighbor_dist)
                ret.append(example)
                self.statistics['#{}_{}'.format(event.label, selected_arg.label)] += 1

                #print('{} {}_{} {}-{}[{}] {}-{}[{}] ||| {}'.format(anchor.label, sentence.docid, sentence.index,
                #                                            anchor.start_char_offset(), anchor.end_char_offset(), '_'.join(token.text for token in anchor.tokens),
                #                                            selected_arg.entity_mention.start_char_offset(), selected_arg.entity_mention.end_char_offset(),
                #                                            '_'.join(token.text for token in selected_arg.entity_mention.tokens),
                #                                            self._sentence_text_with_markup(sentence, anchor,
                #                                                                            selected_arg.entity_mention)))

            #print('')

            # for arg in event.arguments:
            #     """:type arg: nlplingo.text.text_span.EventArgument"""
            #     if arg.label == 'Source':
            #         example = EventArgumentExample(anchor, arg.entity_mention, sentence, self.event_domain, self.params, arg.label)
            #         EventArgumentGenerator._generate_example(example, tokens, self.max_sent_length, self.neighbor_dist)
            #         ret.append(example)

        # get the set of event types for this sentence, or use 'None' if this sentence does not contain any positive event
        # event_types = set()
        # for event in sentence.events:
        #     event_types.add(event.label)
        # if len(event_types) == 0:
        #     event_types.add('None')
        #
        # for event_type in event_types:
        #     example = EventMentionExample(sentence, self.event_domain, self.params, event_type)
        #     self._generate_example(example, sentence.tokens)
        #     ret.append(example)

        return ret

    @classmethod
    def _generate_example(cls, example, tokens):
        """
        :type example: nlplingo.event.event_mention.EventMentionExample
        :type tokens: list[nlplingo.text.text_span.Token]
        :type max_sent_length: int
        """
        event_domain = example.event_domain

        # checks whether the current token is a trigger for multiple event types
        # this doesn't change the state of any variables, it just prints information
        # self.check_token_for_multiple_event_types(token, trigger_ids)

        # TODO if current token is a trigger for multiple event types, event_type_index is only set to 1 event_type_index
        event_type_index = event_domain.get_event_type_index(example.event_type)

        # self.label is a 2-dim matrix [#num_tokens, #event_types]
        example.label[event_type_index] = 1

        # self.vector_data = [ #instances , (max_sent_length + 2*neighbor_dist+1) ]
        # TODO why do we pad on 2*neighbor_dist+1 ??
        cls.assign_vector_data(tokens, example)

    @staticmethod
    def assign_vector_data(tokens, example):
        """Capture the word embeddings, or embeddings index, at each word position in sentence
        :type tokens: list[nlplingo.text.text_span.Token]
        :type example: nlplingo.event.event_mention.EventMentionExample
        """
        for i, token in enumerate(tokens):
            if token and token.has_vector:
                example.vector_data[i] = token.vector_index

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_mention.EventMentionExample]
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec'].append(example.vector_data)
            data_dict['label'].append(example.label)
        return data_dict

class EventMentionExample(object):

    def __init__(self, sentence, event_domain, params, event_type=None):
        """We are given a sentence as the event span, and event_type (present during training)
        :type sentence: nlplingo.text.text_span.Sentence
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        :type event_type: str
        """
        self.sentence = sentence
        self.event_domain = event_domain
        self.event_type = event_type
        self._allocate_arrays(params.get_int('max_sent_length'), params.get_int('embedding.none_token_index'),
                              params.get_string('cnn.int_type'))

    def _allocate_arrays(self, max_sent_length, none_token_index, int_type):
        """Allocates feature vectors and matrices for examples from this sentence
        :type max_sent_length: int
        :type none_token_index: int
        :type int_type: str
        """
        num_labels = len(self.event_domain.event_types)

        # Allocate numpy array for label
        # self.label is a 2 dim matrix: [#instances X #event-types], which I suspect will be 1-hot encoded
        self.label = np.zeros(num_labels, dtype=int_type)

        # Allocate numpy array for data
        self.vector_data = none_token_index * np.ones(max_sent_length, dtype=int_type)




