from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import with_statement

import numpy as np
from ace.ace_docs import AceCorpus

from ace.ace_generate_dataset import AceDataset
from ace.ace_generate_dataset import HengjiPartitionGenerator
from dmcnn.trigger_class_dm_pool_model import loadModel
from generator.trigger import TriggerDataGenerator
from generator.role import RoleDataGenerator

# Load trained trigger and argument role models
triggerModelFilename = 'trigger-classification-run0-epoch5.pickle'
roleModelFilename = 'role-classification-run0-epoch5.pickle'
triggerModel = loadModel(triggerModelFilename)
roleModel = loadModel(roleModelFilename)

# Get instances of word embedding dictionary
prefix = u'ace2005-classification'
tmpDataset = AceDataset()
tmpDataset.load(prefix, _quickDebug=True)
triggerWordEmbedding = tmpDataset.triggerData['tst']['embedding'][()]
roleWordEmbedding = tmpDataset.roleData['tst']['embedding'][()]
# Note triggerWordEmbedding==roleWordEmbedding

aceCorpus = AceCorpus(wordEmbedding=triggerWordEmbedding)
aceCorpus.setPredictionMode()
aceDataset = AceDataset()

triggerGenerator = TriggerDataGenerator(aceCorpus, generateIdentification=False)
roleGenerator = RoleDataGenerator(aceCorpus, generateIdentification=False)

# aceDataset.initialize(HengjiPartitionGenerator(aceCorpus), triggerGenerator, roleGenerator, corpus=AceCorpus)
partitionGenerator = HengjiPartitionGenerator(aceCorpus)
# partitionGenerator.generateCondensedWordEmbedding = False
aceDataset.initialize(partitionGenerator, triggerGenerator, roleGenerator, corpus=aceCorpus)


# partition = u'tst'

# ddecode = Decode(triggerModel, roleModel)
# predictions = ddecode.decode(aceDataset)
# pprint.pprint(predictions)
# print(json.dumps(predictions))

aceDataset.generate(aceDataset.name, generateTriggerData=True, generateRoleData=False, _save=False)
        
triggerPrediction = triggerModel.predict(aceDataset.triggerData)
triggerPredictonIndex = np.argmax(triggerPrediction, axis=1)
positiveTriggerPredictions =  triggerPredictonIndex < len(aceDataset.corpus.domain.getEventTypes())

triggerEventPrediction = [aceDataset.corpus.domain.getEventTypes()[i]
                          for i in triggerPredictonIndex[positiveTriggerPredictions]]

triggerInstanceInfo = aceDataset.triggerData['info'][positiveTriggerPredictions]
segmentSize = aceDataset.generateRoleTestSubset(triggerInstanceInfo, triggerEventPrediction)
rolePrediction = roleModel.predict(aceDataset.roleData)

#ddecode = Decode(triggerModel, roleModel)
#ddecode.generatePredictionDict(triggerEventPrediction, rolePrediction, triggerInstanceInfo, aceDataset)
rolePredictionIndex = np.argmax(rolePrediction, axis=1)

positiveRolePredictions = rolePredictionIndex < len(adataset.corpus.domain.getRoleTypes())
rolePredictionIndex = rolePredictionIndex[positiveRolePredictions]

ridx = 0
for tidx in range(len(triggerEventPrediction)):
    eventType = triggerEventPrediction[tidx]
    print('Event {}'.format(eventType))
    for seg in segmentSize:
        for i in range(seg):
            roleType = adataset.corpus.domain.getRoleTypes()[rolePredictionIndex[ridx]]
            ridx += 1
            if not roleType in adataset.corpus.domain.getEventRoles(eventType):
                print('Role {} not in event {}'.format(roleType, eventType))
        
    
roleInstanceInfo = adataset.roleData['info'][positiveRolePredictions]
instanceTokenIdx2sentTokenIdx = adataset.roleData['tokenIdx'][positiveRolePredictions,:]

rinfoIndex = 0
rinfoIndexMax = roleInstanceInfo.shape[0]
results = []
for i, (eventType, tinfo) in enumerate(zip(triggerEventPrediction, triggerInstanceInfo)):
    docId = tinfo['docId']
    textUnitNo = tinfo['textUnitNo']
    triggerIndex = tinfo['triggerTokenNo']
    while (rinfoIndex < rinfoIndexMax and roleInstanceInfo[rinfoIndex]['textUnitNo'] < textUnitNo):
        rinfoIndex += 1
    while (rinfoIndex < rinfoIndexMax and roleInstanceInfo[rinfoIndex]['triggerTokenNo'] < triggerIndex):
        rinfoIndex += 1
    if rinfoIndex >= rinfoIndexMax:
        break
    if (roleInstanceInfo[rinfoIndex]['textUnitNo'] == textUnitNo
        and roleInstanceInfo[rinfoIndex]['triggerTokenNo'] == triggerIndex):
        corpusDocument = adataset.corpus.getDocument(docId)
        tokens = corpusDocument.getTextUnits()[textUnitNo]
        doc = tokens[0].doc
        first = True
        while (rinfoIndex < rinfoIndexMax 
               and roleInstanceInfo[rinfoIndex]['textUnitNo'] == textUnitNo
               and roleInstanceInfo[rinfoIndex]['triggerTokenNo'] == triggerIndex):
            idxMap = instanceTokenIdx2sentTokenIdx[rinfoIndex]
            if first:
                first = False
                triggerToken = doc[idxMap[triggerIndex]]
                d = {}
                d['extractor'] = 'frames'
                d['frame-type'] = eventType
                d['arguments'] = []
                d['arguments'].append({'start' : triggerToken.idx,
                                       'end' : triggerToken.idx + len(triggerToken),
                                       'label' : 'anchor',
                                       'text' : triggerToken.text})
            rinfo = roleInstanceInfo[rinfoIndex]
            rolePredictionLabel = adataset.corpus.domain.getRoleTypes()[rolePredictionIndex[rinfoIndex]]
            roleIndex = rinfo['roleTokenNo']
            roleToken = doc[idxMap[roleIndex]]

            d['arguments'].append({'start' : roleToken.idx,
                                   'end' : roleToken.idx + len(roleToken),
                                   'label' : rolePredictionLabel,
                                   'text' : roleToken.text})
            rinfoIndex += 1
        results.append(d)




#########################
i = 0
while i < 18341:
    if not roleWordEmbedding.words[i] == triggerWordEmbedding.words[i]:
        break
    i += 1
print(i)

i = 9
while i < 3000:
    if not triggerWordEmbedding.words[triggerWordEmbedding.segmentLocation+i+27] == roleWordEmbedding.words[roleWordEmbedding.segmentLocation+i+28]:
        break
    i += 1
print(i)
