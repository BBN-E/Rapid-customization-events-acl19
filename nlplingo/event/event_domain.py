from __future__ import absolute_import
from __future__ import print_function
from __future__ import unicode_literals


import codecs
import re

from collections import defaultdict


class EventRole(object):
    def __init__(self, label):
        self.label = label
        self.map_to = label


class EventDomain(object):
    """representing the event types and event roles we have in a particular domain"""

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
        self.entity_bio_types = dict()
        self.entity_bio_types_inv = dict()
        self._init_entity_type_indices(entity_types)

        self.domain_name = domain_name

        self.event_type_role = defaultdict(set)
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
        self.entity_bio_types_inv[0] = 'O'
        self.entity_bio_types['O'] = 0
        for i, et in enumerate(entity_types):
            self.entity_types[et] = i
            self.entity_types_inv[i] = et
            self.entity_bio_types_inv[len(self.entity_bio_types)] = et + "_B"
            self.entity_bio_types[et + "_B"] = len(self.entity_bio_types)
            self.entity_bio_types_inv[len(self.entity_bio_types)] = et + "_I"
            self.entity_bio_types[et + "_I"] = len(self.entity_bio_types)
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

    def get_entity_type_from_index(self, index):
        if index in self.entity_types_inv.keys():
            return self.entity_types_inv[index]
        else:
            raise ValueError('Input entity_type_index %d is not in the set of known entity_types: %s' % (index, ','.join(self.entity_types_inv.keys())))


    def get_entity_bio_type_index(self, entity_bio_type):
        if entity_bio_type in self.entity_bio_types.keys():
            return self.entity_bio_types[entity_bio_type]
        else:
            raise ValueError('Input entity_BIO type "%s" is not in the set of known entity_bio_types: %s' % (entity_bio_type, ','.join(self.entity_bio_types.keys())))

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
            raise ValueError('Input event_type_index %d is not in the set of known event_types: %s' % (index, ','.join([str(el) for el in self.event_types_inv.keys()])))

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

    @classmethod
    def read_domain_ontology_file(cls, filepath, domain_name):
        lines = []
        """:type: list[str]"""
        with codecs.open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                lines.append(line.strip())

        event_type_role = defaultdict(set)
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

    def to_string(self):
        ret = []
        ret.append('Event-types:')
        for et, et_index in self.event_types.items():
            ret.append('{} -> {}'.format(et, str(et_index)))
        ret.append('Event-type-role:')
        for et in self.event_type_role:
            for role in self.event_type_role[et]:
                ret.append('{} {}->{}'.format(et, role.label, role.map_to))
        return '\n'.join(ret)


class AceDomain(EventDomain):
    EVENT_TYPES = [ 'Life.Be-Born', 'Life.Die', 'Life.Marry', 'Life.Divorce', 'Life.Injure',
                    'Transaction.Transfer-Ownership', 'Transaction.Transfer-Money',
                    'Movement.Transport',
                    'Business.Start-Org', 'Business.End-Org', 'Business.Declare-Bankruptcy', 'Business.Merge-Org',
                    'Conflict.Attack', 'Conflict.Demonstrate',
                    'Contact.Meet', 'Contact.Phone-Write',
                    'Personnel.Start-Position', 'Personnel.End-Position', 'Personnel.Nominate', 'Personnel.Elect',
                    'Justice.Arrest-Jail', 'Justice.Release-Parole', 'Justice.Charge-Indict', 'Justice.Trial-Hearing',
                    'Justice.Sue', 'Justice.Convict', 'Justice.Sentence', 'Justice.Fine', 'Justice.Execute',
                    'Justice.Extradite', 'Justice.Acquit', 'Justice.Pardon', 'Justice.Appeal']
    EVENT_ROLES = [ 'Person', 'Place', 'Buyer', 'Seller', 'Beneficiary', 'Price',
                    'Artifact', 'Origin', 'Destination', 'Giver', 'Recipient', 'Money',
                    'Org', 'Agent', 'Victim', 'Instrument', 'Entity', 'Attacker', 'Target',
                    'Defendant', 'Adjudicator', 'Prosecutor', 'Plaintiff', 'Crime',
                    'Position', 'Sentence', 'Vehicle', 'Time-Within', 'Time-Starting',
                    'Time-Ending', 'Time-Before', 'Time-After', 'Time-Holds', 'Time-At-Beginning', 'Time-At-End']
    ENTITY_TYPES = ['FAC', 'PER', 'LOC', 'GPE', 'ORG', 'WEA', 'VEH', 'Sentence', 'Job-Title', 'Crime', 'Contact-Info',
                    'Numeric', 'Time']

    def __init__(self):
        EventDomain.__init__(self, self.EVENT_TYPES, self.EVENT_ROLES, self.ENTITY_TYPES)

    def domain(self):
        return 'ACE'


class CyberDomain(EventDomain):
    EVENT_TYPES = ['CyberAttack', 'Vulnerability', 'Exploit']
    EVENT_ROLES = ['CyberAttackType', 'Date', 'ExploitType', 'ImpactType', 'Method', 'Name', 'Software',
                   'Source', 'Target', 'VulnerabilityType']
    ENTITY_TYPES = ['Organization', 'OrganizationDesc', 'Person', 'PersonDesc', 'GPE', 'GPEDesc', 'HackableThing',
                    'HackableThingDesc', 'ProductVendor', 'ProductVendorDesc', 'ExploitName', 'ExploitDesc',
                    'AttackType', 'VulnerabilityType', 'ImpactType']

    def __init__(self):
        EventDomain.__init__(self, self.EVENT_TYPES, self.EVENT_ROLES, self.ENTITY_TYPES)

    def domain(self):
        return 'CYBER'


class PrecursorDomain(EventDomain):
    EVENT_TYPES = ['BlockTransaction', 'Censor', 'CloseAccount', 'FreezeFund', 'Layoff', 'Scandal']
    EVENT_ROLES = ['Source','Target','Time']
    ENTITY_TYPES = ['Organization', 'OrganizationDesc', 'Person', 'PersonDesc', 'GPE', 'GPEDesc']

    def __init__(self):
        EventDomain.__init__(self, self.EVENT_TYPES, self.EVENT_ROLES, self.ENTITY_TYPES)

    def domain(self):
        return 'PRECURSOR'


class AcePrecursorDomain(EventDomain):
    EVENT_TYPES = [ 'Life.Be-Born', 'Life.Die', 'Life.Marry', 'Life.Divorce', 'Life.Injure',
                    'Transaction.Transfer-Ownership', 'Transaction.Transfer-Money',
                    'Movement.Transport',
                    'Business.Start-Org', 'Business.End-Org', 'Business.Declare-Bankruptcy', 'Business.Merge-Org',
                    'Conflict.Attack', 'Conflict.Demonstrate',
                    'Contact.Meet', 'Contact.Phone-Write',
                    'Personnel.Start-Position', 'Personnel.End-Position', 'Personnel.Nominate', 'Personnel.Elect',
                    'Justice.Arrest-Jail', 'Justice.Release-Parole', 'Justice.Charge-Indict', 'Justice.Trial-Hearing',
                    'Justice.Sue', 'Justice.Convict', 'Justice.Sentence', 'Justice.Fine', 'Justice.Execute',
                    'Justice.Extradite', 'Justice.Acquit', 'Justice.Pardon', 'Justice.Appeal',
                    'BlockTransaction', 'Censor', 'CloseAccount', 'FreezeFund', 'Layoff', 'Scandal']
    EVENT_ROLES = [ 'Person', 'Place', 'Buyer', 'Seller', 'Beneficiary', 'Price',
                    'Artifact', 'Origin', 'Destination', 'Giver', 'Recipient', 'Money',
                    'Org', 'Agent', 'Victim', 'Instrument', 'Entity', 'Attacker', 'Target',
                    'Defendant', 'Adjudicator', 'Prosecutor', 'Plaintiff', 'Crime',
                    'Position', 'Sentence', 'Vehicle', 'Time-Within', 'Time-Starting',
                    'Time-Ending', 'Time-Before', 'Time-After', 'Time-Holds', 'Time-At-Beginning', 'Time-At-End']
    ENTITY_TYPES = ['FAC', 'PER', 'LOC', 'GPE', 'ORG', 'WEA', 'VEH', 'Sentence', 'Job-Title', 'Crime', 'Contact-Info',
                    'Numeric', 'Time']

    def __init__(self):
        EventDomain.__init__(self, self.EVENT_TYPES, self.EVENT_ROLES, self.ENTITY_TYPES)

    def domain(self):
        return 'ACE-PRECURSOR'


class UIDomain(EventDomain):
    EVENT_TYPES = ['Conflict.Attack']
    EVENT_ROLES = ['Attacker', 'Target', 'Instrument', 'Place', 'Time']
    #ENTITY_TYPES = ['Organization', 'OrganizationDesc', 'Person', 'PersonDesc', 'GPE', 'GPEDesc']
    ENTITY_TYPES = []

    def __init__(self):
        EventDomain.__init__(self, self.EVENT_TYPES, self.EVENT_ROLES, self.ENTITY_TYPES)

    def domain(self):
        return 'UI'


