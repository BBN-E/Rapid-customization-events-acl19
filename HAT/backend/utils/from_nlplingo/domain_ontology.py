import collections,codecs,re

class EventRole(object):
    def __init__(self, label):
        self.label = label
        self.map_to = label


class EventDomain(object):
    """An abstract class representing the event types and event roles we have in a particular domain"""
    #__metaclass__ = ABCMeta

    def __init__(self, event_types, event_roles, entity_types, domain_name=None):
        """
        :type event_types: list[str]
        :type event_roles: list[str]
        """
        self.event_types = dict()
        self.event_types_inv = dict()
        self._init_event_type_indices(event_types)

        self.event_roles = dict()
        self.event_roles_inv = dict()
        self._init_event_role_indices(event_roles)

        self.entity_types = dict()
        self.entity_types_inv = dict()
        self._init_entity_type_indices(entity_types)

        self.domain_name = domain_name

        self.event_type_role = collections.defaultdict(set)
        """:type: dict[str, set[EventRole]]"""

    #@abstractmethod
    #def domain(self):
    #    """Returns a string representing the current event domain"""
    #    pass

    def event_type_in_domain(self, et):
        return et in self.event_types.keys()

    def event_type_role_in_domain(self, et, er):
        if et in self.event_type_role:
            for role in self.event_type_role[et]:
                if er == role.label:
                    return True
            return False
        else:
            return False

    # def constraint_event_role_to_domain(self, event):
    #     """:type event: nlplingo.text.text_theory.Event"""
    #     et = event.label
    #     new_args = [arg for arg in event.arguments if self.event_type_role_in_domain(et, arg.label)]
    #     event.arguments = new_args

    # def apply_mappings(self, event):
    #     """Currently we only have role mappings"""
    #     et = event.label
    #     for arg in event.arguments:
    #         for role in self.event_type_role[et]:
    #             if role.label == arg.label:
    #                 arg.label = role.map_to
    #                 break

    def _init_entity_type_indices(self, entity_types):
        """
        :type entity_types: list[str]
        """
        for i, et in enumerate(entity_types):
            self.entity_types[et] = i
            self.entity_types_inv[i] = et
        self.entity_types['None'] = len(entity_types)
        self.entity_types_inv[len(entity_types)] = 'None'

    def _init_event_type_indices(self, event_types):
        """
        :type event_types: list[str]
        """
        for i, et in enumerate(event_types):
            self.event_types[et] = i
            self.event_types_inv[i] = et
        self.event_types['None'] = len(event_types)
        self.event_types_inv[len(event_types)] = 'None'


    def _init_event_role_indices(self, event_roles):
        """
        :type event_roles: list[str]
        """
        for i, er in enumerate(event_roles):
            self.event_roles[er] = i
            self.event_roles_inv[i] = er
        self.event_roles['None'] = len(event_roles)
        self.event_roles_inv[len(event_roles)] = 'None'

    def get_entity_type_index(self, entity_type):
        if entity_type in self.entity_types.keys():
            return self.entity_types[entity_type]
        else:
            raise ValueError('Input entity_type "%s" is not in the set of known entity_types: %s' % (entity_type, ','.join(self.entity_types.keys())))

    def get_event_type_index(self, event_type):
        """
        :type event_type: str
        Returns:
            int
        """
        if event_type in self.event_types.keys():
            return self.event_types[event_type]
        else:
            raise ValueError('Input event_type "%s" is not in the set of known event_types: %s' % (event_type, ','.join(self.event_types.keys())))

    def get_event_type_from_index(self, index):
        if index in self.event_types_inv.keys():
            return self.event_types_inv[index]
        else:
            raise ValueError('Input event_type_index %d is not in the set of known event_types: %s' % (index, ','.join(self.event_types_inv.keys())))

    def get_event_role_index(self, event_role):
        """
        :type event_role: str
        Returns:
            int
        """

        if event_role in self.event_roles:
            return self.event_roles[event_role]
        else:
            raise ValueError('Input event_role "%s" is not in the set of known event_roles: %s' % (event_role, ','.join(self.event_roles.keys())))

    def get_event_role_from_index(self, index):
        if index in self.event_roles_inv.keys():
            return self.event_roles_inv[index]
        else:
            raise ValueError('Input event_role_index %d is not in the set of known event_roles: %s' % (index, ','.join(self.event_roles_inv.keys())))

def read_domain_ontology_file(filepath, domain_name):
    lines = []
    """:type: list[str]"""
    with codecs.open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            lines.append(line.strip())

    event_type_role = collections.defaultdict(set)
    event_types = set()
    roles = set()
    entity_types = set()
    entity_subtypes = set()
    i = 0
    et = None
    while i < len(lines):
        line = lines[i]
        if line.startswith('<Event type='):
            et = re.search(r' type="(.*?)"', line).group(1)
            event_types.add(et)
        elif line.startswith('<Role'):
            role = re.search(r'>(.*)</Role>', line).group(1)
            roles.add(role)
            if 'map-from=' in line:
                map_from = re.search(r'map-from="(.*?)"', line).group(1)
            else:
                map_from = role

            er = EventRole(map_from)
            er.map_to = role
            event_type_role[et].add(er)
        elif line.startswith('<Entity '):
            if ' type=' in line:
                entity_types.add(re.search(r' type="(.*?)"', line).group(1))
            elif ' subtype=' in line:
                entity_subtypes.add(re.search(r' subtype="(.*?)"', line).group(1))

        i += 1

    event_domain = EventDomain(sorted(list(event_types)), sorted(list(roles)), sorted(list(
        entity_types)), domain_name)
    event_domain.event_type_role = event_type_role
    return event_domain