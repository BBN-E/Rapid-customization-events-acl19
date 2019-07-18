
import re
import yaml
from event_trigger_data_management.model import EventMention

class Serializer(object):
    def serialize(self):
        raise NotImplementedError
