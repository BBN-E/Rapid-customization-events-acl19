
import re

class Parameters(object):

    def __init__(self, config_file=None, overrides=None):
        self.params = {}

        if overrides:
            for key in overrides:
                self.params[key] = overrides[key]

        if config_file is not None:
            f = open(config_file, 'r')
            for line in f:
                line = line.strip()

                if len(line) == 0 or line.startswith('#'):
                    continue

                index = line.find(':')
                key = line[0:index].strip()
                value = line[index+1:].strip()

                if overrides and key in overrides:
                    continue

                while re.search(r'%(.*?)%', value):
                    re_key = re.search(r'%(.*?)%', value).group(1)
                    if re_key in self.params:
                        re_value = self.params[re_key]
                        value = re.sub(r'%'+re_key+'%', re_value, value)
                    else:
                        raise ValueError('Cannot find re_key {} in params file'.format(re_key))

                self.params[key] = value

            f.close()

    def has_key(self, key):
        return key in self.params

    def get_string(self, key):
        """
        :return: str
        :rtype: str
        """
        return self.params.get(key)

    def get_utf8_string(self, key):
        """
        :return: str
        """
        return self.params.get(key).encode('utf-8')

    def get_int(self, key):
        """
        :return: int
        :rtype: int
        """
        return int(self.params.get(key))

    def get_float(self, key):
        """
        :return: float
        """
        return float(self.params.get(key))

    def get_list(self, key):
        """
        Returns:
            list[str]
        """
        if key in self.params:
            return self.params.get(key).split(',')
        else:
            return None

    def get_boolean(self, key):
        if key in self.params:
            return self.to_boolean(self.params.get(key))
        else:
            return None

    def to_boolean(self, v):
        if v == 'True' or v == 'true':
            return True
        else:
            return False

    def print_params(self):
        for k in sorted(self.params):
            print('%s: %s' % (k, self.params.get(k)))


if __name__ == "__main__":
    params = Parameters('ner.params')
    params.print_params()




