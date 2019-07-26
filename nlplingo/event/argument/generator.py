
from collections import defaultdict

from nlplingo.text.text_span import Anchor
from nlplingo.common.utils import IntPair
from nlplingo.event.trigger.generator import EventTriggerExampleGenerator
from nlplingo.event.argument.example import EventArgumentExample

class EventArgumentExampleGenerator(object):
    verbosity = 0

    def __init__(self, event_domain, params, extractor_params, hyper_params):
        """
        :type event_domain: nlplingo.event.event_domain.EventDomain
        :type params: dict
        :type extractor_params: dict
        :type hyper_params: nlplingo.nn.extractor.HyperParameters
        """
        self.event_domain = event_domain
        self.params = params
        self.extractor_params = extractor_params
        self.hyper_params = hyper_params
        self.statistics = defaultdict(int)

    def generate(self, docs, feature_generator, triggers=None):
        """
        +1
        :type docs: list[nlplingo.text.text_theory.Document]
        :type feature_generator: nlplingo.event.argument.feature.EventArgumentFeatureGenerator
        :type triggers: defaultdict(list[nlplingo.event.trigger.example.EventTriggerExample])
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.argument.example.EventArgumentExample]"""

        for doc in docs:
            if triggers is not None:
                doc_triggers = triggers[doc.docid]
                """:type: list[nlplingo.event.trigger.example.EventTriggerExample]"""

                # organize the doc_triggers by sentence number
                sent_triggers = defaultdict(list)
                for trigger in doc_triggers:
                    sent_triggers[trigger.sentence.index].append(trigger)

                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent, feature_generator, trigger_egs=sent_triggers[sent.index]))
            else:
                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent, feature_generator))

        for k, v in self.statistics.items():
            print('EventArgumentExampleGenerator stats, {}:{}'.format(k, v))

        return examples

    @staticmethod
    def get_event_role(anchor, entity_mention, events):
        """
        +1
        If the given (anchor, entity_mention) is found in the given events, return role label, else return 'None'
        :type anchor: nlplingo.text.text_span.Anchor
        :type entity_mention: nlplingo.text.text_span.EntityMention
        :type events: list[nlplingo.text.text_theory.Event]
        """

        for event in events:
            for a in event.anchors:
                if anchor.start_char_offset() == a.start_char_offset() and anchor.end_char_offset() == a.end_char_offset():
                    role = event.get_role_for_entity_mention(entity_mention)
                    if role != 'None':
                        return role
        return 'None'

    def _generate_sentence(self, sentence, feature_generator, trigger_egs=None):
        """
        +1
        We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        :type feature_generator: nlplingo.event.argument.feature.EventArgumentFeatureGenerator
        :type trigger_egs: list[nlplingo.event.trigger.example.EventTriggerExample]
        """
        # skip multi-token triggers, args that do not have embeddings, args that overlap with trigger
        ret = []
        """:type: list[nlplingo.event.argument.example.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.hyper_params.max_sentence_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        if trigger_egs is not None:
            for trigger_index, eg in enumerate(trigger_egs):
                anchor_id = '{}-s{}-t{}'.format(sentence.docid, sentence.index, trigger_index)
                anchor = Anchor(anchor_id, IntPair(eg.anchor.start_char_offset(), eg.anchor.end_char_offset()),
                                eg.anchor.text, eg.event_type)
                anchor.with_tokens(eg.anchor.tokens)

                for em in sentence.entity_mentions:
                    role = 'None'
                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params,
                                                       self.extractor_params, feature_generator.features, self.hyper_params, role)
                        feature_generator.generate_example(example, sentence.tokens, self.hyper_params)
                        ret.append(example)
        else:
            for event in sentence.events:
                for anchor in event.anchors:
                    if anchor.head().pos_category() in EventTriggerExampleGenerator.trigger_pos_category:
                        for em in sentence.entity_mentions:
                            role = event.get_role_for_entity_mention(em)
                            self.statistics['#Event-Role {}'.format(role)] += 1
                            # if spans_overlap(anchor, em):
                            #     print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
                            # else:
                            #     if role != 'None':
                            #         self.statistics['number_positive_argument'] += 1
                            #     example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
                            #     self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                            #     ret.append(example)
                            if role != 'None':
                                self.statistics['number_positive_argument'] += 1
                            if em.coarse_label() in self.event_domain.entity_types.keys():
                                example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params,
                                                               self.extractor_params, feature_generator.features, self.hyper_params, role)
                                feature_generator.generate_example(example, sentence.tokens, self.hyper_params)
                                ret.append(example)

        return ret

    @staticmethod
    def examples_to_data_dict(examples, features):
        """
        +1
        :type examples: list[nlplingo.event.argument.example.EventArgumentExample
        :type features: nlplingo.event.argument.feature.EventArgumentFeature
        """
        data_dict = defaultdict(list)
        for example in examples:
            example_data = example.to_data_dict(features)
            for k, v in example_data.items():
                data_dict[k].append(v)

        return data_dict


class EventArgumentExampleGeneratorBySentence(EventArgumentExampleGenerator):
    def __init__(self, event_domain, params, extractor_params):
        super(EventArgumentExampleGeneratorBySentence, self).__init__(event_domain, params, extractor_params)

    def generate(self, docs, triggers=None):
        """
        :type docs: list[nlplingo.text.text_theory.Document]
        :type triggers: defaultdict(list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        self.statistics.clear()

        examples = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        for doc in docs:
            if triggers is not None:
                doc_triggers = triggers[doc.docid]
                """:type: list[nlplingo.event.event_trigger.EventTriggerExample]"""
                print('EventArgumentExampleGenerator.generate(): doc.docid={}, len(doc_triggers)={}'.format(doc.docid, len(doc_triggers)))

                # organize the doc_triggers by sentence number
                sent_triggers = defaultdict(list)
                for trigger in doc_triggers:
                    sent_triggers[trigger.sentence.index].append(trigger)

                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent, trigger_egs=sent_triggers[sent.index]))
            else:
                for sent in doc.sentences:
                    examples.extend(self._generate_sentence(sent))

        for k, v in self.statistics.items():
            print('EventArgumentExampleGenerator stats, {}:{}'.format(k,v))

        return examples

    def _generate_sentence(self, sentence, trigger_egs=None):
        """We could optionally be given a list of anchors, e.g. predicted anchors
        :type sentence: nlplingo.text.text_span.Sentence
        :type trigger_egs: list[nlplingo.event.event_trigger.EventTriggerExample]
        """
        # skip multi-token triggers, args that do not have embeddings, args that overlap with trigger
        ret = []
        """:type: list[nlplingo.event.event_argument.EventArgumentExample]"""

        if sentence.number_of_tokens() < 1:
            return ret
        if sentence.number_of_tokens() > self.max_sent_length:
            print('Skipping overly long sentence of {} tokens'.format(sentence.number_of_tokens()))
            return ret

        if trigger_egs is not None:
            for trigger_index, eg in enumerate(trigger_egs):
                anchor_id = '{}-s{}-t{}'.format(sentence.docid, sentence.index, trigger_index)
                anchor = Anchor(anchor_id, IntPair(eg.anchor.start_char_offset(), eg.anchor.end_char_offset()), eg.anchor.text, eg.event_type)
                anchor.with_tokens(eg.anchor.tokens)

                for em in sentence.entity_mentions:
                    role = 'None'

                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                        self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                        ret.append(example)
        else:
            event_label_to_event_anchor = defaultdict(list)
            for event in sentence.events:

                for anchor in event.anchors:
                    if anchor.head().pos_category() in EventTriggerExampleGenerator.trigger_pos_category:
                        event_label_to_event_anchor[event.label].append((event, anchor))
            print('Sentence')
            for event_label, events_anchors in event_label_to_event_anchor.items():
                print('Event {}'.format(event_label))
                em_to_role = ['None'] * len(sentence.entity_mentions)
                for event, anchor in events_anchors:
                    for i, em in enumerate(sentence.entity_mentions):
                        role = event.get_role_for_entity_mention(em)
                        if role != 'None':
                            em_to_role[i] = role

                for i, em in enumerate(sentence.entity_mentions):
                    role = em_to_role[i]
                    anchor = events_anchors[0][1]
                    self.statistics['#Event-Role {}'.format(role)] += 1
                    print('{} {}'.format(i, role))
                    # if spans_overlap(anchor, em):
                    #     print('Refusing to consider overlapping anchor [%s] and entity_mention [%s] as EventArgumentExample' % (anchor.to_string(), em.to_string()))
                    # else:
                    #     if role != 'None':
                    #         self.statistics['number_positive_argument'] += 1
                    #     example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, role)
                    #     self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                    #     ret.append(example)
                    if role != 'None':
                        self.statistics['number_positive_argument'] += 1
                    if em.coarse_label() in self.event_domain.entity_types.keys():
                        example = EventArgumentExample(anchor, em, sentence, self.event_domain, self.params, self.extractor_params, role)
                        self._generate_example(example, sentence.tokens, self.max_sent_length, self.neighbor_dist, self.do_dmcnn)
                        ret.append(example)
            print('End Sentence')

        return ret
