from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import codecs
import random
from collections import defaultdict

pair_labels = {'DIFFERENT':0, 'SAME':1}

class EventPairData(object):
    def __init__(self, trigger_examples, pair_examples, data, data_list, label):
        """
        :type trigger_examples: list[nlplingo.event.event_trigger.EventTriggerExample]
        :type pair_examples: list[nlplingo.event.event_pair.EventPairExample]
        :type data: defaultdict[str, list[numpy.ndarray]]
        :type data_list: list[numpy.ndarray]
        :type label: numpy.ndarray
        """
        self.trigger_examples = trigger_examples
        self.pair_examples = pair_examples
        self.data = data
        self.data_list = data_list
        self.label = label


class EventPairGenerator(object):

    def __init__(self, event_domain, params, word_embeddings):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: nlplingo.common.parameters.Parameters
        :type word_embeddings: nlplingo.embeddings.word_embeddings.WordEmbedding
        """
        self.word_embeddings = word_embeddings
        self.event_domain = event_domain
        self.params = params
        self.max_sent_length = params.get_int('max_sent_length')
        self.max_samples = params.get_int('max_samples')

        # these are the labels that we allow during training. Other labels are treated as 'None'
        if params.has_key('train_labels'):
            self.train_labels = set(params.get_list('train_labels'))
        else:
            self.train_labels = None

        self.statistics = defaultdict(int)

    def generate_train(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        self.statistics.clear()
        number_positive = 0
        number_negative = 0

        ret = []
        """:type: list[nlplingo.event.event_pair.EventPairExample]"""

        # first, let's group examples by event_type
        example_by_event_type = defaultdict(list)
        """:type: dict[str, list[nlpling.event.event_trigger.EventTriggerExample]]"""

        for example in examples:
            event_type = example.event_type
            if self.train_labels is not None and event_type not in self.train_labels:
                event_type = 'None'
            if event_type != 'None':
                example_by_event_type[event_type].append(example)

        event_types = [et for et in sorted(example_by_event_type.keys())]

        # now let's pair up the examples. Intra-class are positives, Inter-class are negatives
        for label_index, label in enumerate(event_types):
            assert label != 'None'

            label_examples = example_by_event_type.get(label)
            self.statistics['#{}'.format(label)] = len(label_examples)

            # first, the positive examples: intra-class for all positive classes
            candidates = []
            for i in range(len(label_examples) - 1):
                for j in range(i + 1, len(label_examples)):
                    candidates.append(EventPairExample(self.params, label_examples[i], label_examples[j], 'SAME'))
                    self.statistics['P #{}'.format(label)] += 1

            if len(candidates) > self.max_samples:
                selected_candidates = random.sample(candidates, self.max_samples)
            else:
                selected_candidates = candidates
            number_positive += len(selected_candidates)
            ret.extend(selected_candidates)
            self.statistics['S-P #{}'.format(label)] += len(selected_candidates)

            # now, the negative examples: inter-class over pairs of positive classes
            for other_index in range(label_index+1, len(event_types)):
                other_label = event_types[other_index]
                assert other_label != 'None'
                other_examples = example_by_event_type.get(other_label)
                candidates = []
                for eg1 in label_examples:
                    for eg2 in other_examples:
                        candidates.append(EventPairExample(self.params, eg1, eg2, 'DIFFERENT'))
                        self.statistics['N #{}_{}'.format(label, other_label)] += 1

                if len(candidates) > self.max_samples:
                    selected_candidates = random.sample(candidates, self.max_samples)
                else:
                    selected_candidates = candidates
                number_negative += len(selected_candidates)
                ret.extend(selected_candidates)
                self.statistics['S-N #{}_{}'.format(label, other_label)] += len(selected_candidates)

        for k in sorted(self.statistics.keys()):
            print('EventPairGenerator stats, {}:{}'.format(k, self.statistics.get(k)))
        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))
        return ret

    # def generate(self, examples):
    #     """
    #     :type examples: list[event.event_mention.EventMentionExample]
    #     """
    #     self.statistics.clear()
    #     number_positive = 0
    #     number_negative = 0
    #
    #     ret = []
    #     """:type: list[event.event_pair.EventPairExample]"""
    #
    #     for eg in examples:
    #         ret.append(EventPairExample(self.params, eg, eg, 'SAME'))
    #         number_positive += 1
    #
    #     for i in range(len(examples)-1):
    #         ret.append(EventPairExample(self.params, examples[i], examples[i+1], 'DIFFERENT'))
    #         number_negative += 1
    #
    #     print('#Positives = {}'.format(number_positive))
    #     print('#Negatives = {}'.format(number_negative))
    #     return ret

    def generate_test(self, examples):
        """
        :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        number_positive = 0
        number_negative = 0

        ret = []
        """:type: list[nlplingo.event.event_pair.EventPairExample]"""

        for i, example in enumerate(examples):
            eg1 = examples[i]
            for j in range(i+1, len(examples)):
                eg2 = examples[j]

                if eg1.event_type == eg2.event_type:
                    if eg1.event_type != 'None':    # since given two 'None'-'None', we cannot say whether they are same or different
                        ret.append(EventPairExample(self.params, eg1, eg2, 'SAME'))
                        number_positive += 1
                else:
                    ret.append(EventPairExample(self.params, eg1, eg2, 'DIFFERENT'))
                    number_negative += 1

                # if eg1.event_type == eg2.event_type:
                #     if eg1.anchor.label not in self.train_labels and (eg1.anchor.label != 'None'):
                #         # we are only interested in testing the novel type not seen in training data
                #         ret.append(EventPairExample(self.params, eg1, eg2, 'SAME'))
                #         number_positive += 1
                # else:
                #     ret.append(EventPairExample(self.params, eg1, eg2, 'DIFFERENT'))
                #     number_negative += 1

        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))

        return ret

    def generate_cross_product(self, egs1, egs2):
        """
        :type egs1: list[nlplingo.event.event_trigger.EventTriggerExample]
        :type egs2: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        number_positive = 0
        number_negative = 0

        ret = []
        """:type: list[nlplingo.event.event_pair.EventPairExample]"""

        for eg1 in egs1:
            for eg2 in egs2:

                if eg1.token == eg2.token and eg1.sentence.docid == eg2.sentence.docid:
                    continue

                if eg1.event_type == eg2.event_type:
                    if eg1.event_type != 'None':  # since given two 'None'-'None', we cannot say whether they are same or different
                        ret.append(EventPairExample(self.params, eg1, eg2, 'SAME'))
                        number_positive += 1
                else:
                    ret.append(EventPairExample(self.params, eg1, eg2, 'DIFFERENT'))
                    number_negative += 1

        print('#Positives = {}'.format(number_positive))
        print('#Negatives = {}'.format(number_negative))
        return ret

    def examples_to_data_dict(self, examples):
        """
        :type examples: list[nlplingo.event.event_pair.EventPairExample]
        """
        data_dict = defaultdict(list)
        for example in examples:
            data_dict['word_vec1'].append(example.vector_data1)
            data_dict['word_vec2'].append(example.vector_data2)
            data_dict['pos_data1'].append(example.pos_data1)
            data_dict['pos_data2'].append(example.pos_data2)

            #data_dict['word_cvec1'].append(example.cvector_data1)
            #data_dict['word_cvec2'].append(example.cvector_data2)

            #data_dict['dep_vec1'].append(example.dep_data1)
            #data_dict['dep_vec2'].append(example.dep_data2)

            data_dict['label'].append(example.label)
        return data_dict

class EventPairExample(object):

    def __init__(self, params, eg1, eg2, label_string):
        """We are given a sentence as the event span, and event_type (present during training)
        :type params: common.parameters.Parameters
        :type eg1: nlplingo.event.event_trigger.EventTriggerExample
        :type eg2: nlplingo.event.event_trigger.EventTriggerExample
        :type label_string: 'SAME' or 'DIFFERENT'
        """
        self.label_string = label_string
        self.eg1 = eg1
        self.eg2 = eg2
        #self._allocate_arrays(params.get_int('max_sent_length'), params.get_int('embedding.none_token_index'),
        #                      params.get_string('cnn.int_type'))
        self.label = pair_labels[label_string]
        self.vector_data1 = eg1.vector_data
        self.vector_data2 = eg2.vector_data

        #self.cvector_data1 = eg1.cvector_data
        #self.cvector_data2 = eg2.cvector_data

        self.pos_data1 = eg1.pos_data
        self.pos_data2 = eg2.pos_data

        #self.dep_data1 = eg1.dep_data
        #self.dep_data2 = eg2.dep_data

    # def _allocate_arrays(self, max_sent_length, none_token_index, int_type):
    #     """Allocates feature vectors and matrices for examples from this sentence
    #     :type max_sent_length: int
    #     :type none_token_index: int
    #     :type int_type: str
    #     """
    #     num_labels = 2
    #
    #     # Allocate numpy array for label
    #     # self.label is a 2 dim matrix: [#instances X #event-types], which I suspect will be 1-hot encoded
    #     #self.label = np.zeros(num_labels, dtype=int_type)
    #     self.label = None
    #
    #     # Allocate numpy array for data
    #     self.vector_data1 = none_token_index * np.ones(max_sent_length, dtype=int_type)
    #     self.vector_data2 = none_token_index * np.ones(max_sent_length, dtype=int_type)

def print_pair_predictions(pair_examples, predictions, outfilepath):
    """
    :type pair_examples: list[nlplingo.event.event_pair.EventPairExample]
    :type predictions: list[numpy.ndarray]
    """
    assert len(pair_examples) == len(predictions)

    o = codecs.open(outfilepath, 'w', encoding='utf-8')
    for i, eg in enumerate(pair_examples):
        eg1_token_info = '{}:{}-{}:{}'.format(eg.eg1.sentence.docid, eg.eg1.token.start_char_offset(), eg.eg1.token.end_char_offset(), eg.eg1.token.text)
        eg2_token_info = '{}:{}-{}:{}'.format(eg.eg2.sentence.docid, eg.eg2.token.start_char_offset(), eg.eg2.token.end_char_offset(), eg.eg2.token.text)
        et1 = eg.eg1.event_type
        et2 = eg.eg2.event_type
        prob = predictions[i][0]    # type(predictions[i]=<type 'numpy.ndarray'>, predictions[i].shape=(1,)

        o.write('{} {} {} {} {}\n'.format(prob, et1, et2, eg1_token_info, eg2_token_info))
    o.close()


