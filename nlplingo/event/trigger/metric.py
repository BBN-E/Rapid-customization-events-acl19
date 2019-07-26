from collections import defaultdict

import numpy as np

def get_recall_misses(prediction, label, none_class_index, event_domain, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    """
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    assert len(label_arg_max) == len(pred_arg_max)
    assert len(label_arg_max) == len(examples)

    misses = defaultdict(int)
    for i in range(len(label_arg_max)):
        if label_arg_max[i] != none_class_index:
            et = event_domain.get_event_type_from_index(label_arg_max[i])
            if pred_arg_max[i] != label_arg_max[i]:  # recall miss
                eg = examples[i]
                anchor_string = '_'.join(token.text.lower() for token in eg.anchor.tokens)
                misses['{}\t({})'.format(et, anchor_string)] += 1
    return misses


def get_precision_misses(prediction, label, none_class_index, event_domain, examples):
    """
    :type event_domain: nlplingo.event.event_domain.EventDomain
    :type examples: list[nlplingo.event.event_trigger.EventTriggerExample]
    """
    label_arg_max = np.argmax(label, axis=1)
    pred_arg_max = np.argmax(prediction, axis=1)
    assert len(label_arg_max) == len(pred_arg_max)
    assert len(label_arg_max) == len(examples)

    misses = defaultdict(int)
    for i in range(len(pred_arg_max)):
        if pred_arg_max[i] != none_class_index:
            et = event_domain.get_event_type_from_index(pred_arg_max[i])
            if pred_arg_max[i] != label_arg_max[i]:  # precision miss
                eg = examples[i]
                anchor_string = '_'.join(token.text.lower() for token in eg.anchor.tokens)
                misses['{}\t({})'.format(et, anchor_string)] += 1
    return misses
