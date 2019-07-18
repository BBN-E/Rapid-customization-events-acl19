import collections

Trigger_Item_Model = collections.namedtuple('Trigger_Item_Model',['POS','text','wn_sense'])


class OntologyTreeNode(object):
    def __init__(self,**kwargs):
        self.__dict__.update(kwargs)

class InternalOntologyTreeNode(OntologyTreeNode):
    pass

class HumeOntologyTreeNode(OntologyTreeNode):
    pass
