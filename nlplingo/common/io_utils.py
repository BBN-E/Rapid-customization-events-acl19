import codecs
import json

class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'reprJSON'):
            return obj.reprJSON()
        else:
            return json.JSONEncoder.default(self, obj)


class JsonObject:
    # Serialization helper
    def reprJSON(self):
        d = dict()
        for a, v in self.__dict__.items():
            if v is None:
                continue
            if (hasattr(v, "reprJSON")):
                d[a] = v.reprJSON()
            else:
                d[a] = v
        return d


def read_file_to_set(filename):
    ret = set()
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            ret.add(line.strip())
    return ret


def read_file_to_list(filename):
    with codecs.open(filename, 'r', encoding='utf-8') as f:
        ret = [line.strip() for line in f]
    return ret

def write_list_to_file(lines, filepath):
    with codecs.open(filepath, 'w', encoding='utf-8') as o:
        for line in lines:
            o.write(line)
            o.write('\n')

def safeprint(s):
    print(s.encode('ascii', 'ignore'))