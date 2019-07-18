
import os, sys,json
current_script_path = __file__
sys.path.append(os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir)))
project_root = os.path.realpath(os.path.join(current_script_path,os.path.pardir,os.path.pardir))

try:
    # Python 3.x
    from urllib.parse import quote_plus
except ImportError:
    # Python 2.x
    from urllib import quote_plus

class Config(object):
    def __init__(self):
        pass
    def FOLDER_SESSION(self,session):
        return os.path.join(self.BACKEND_RUNTIME_FOLDER,session)

    def restore_to_dict(self):
        def revert_list_into_list(root):
            ret = list()
            for i in root:
                if isinstance(i, list):
                    ret.append(revert_list_into_list(root))
                elif isinstance(i, Config):
                    ret.append(i.restore_to_dict())
                else:
                    ret.append(i)
            return ret
        ret = dict()
        for k,v in self.__dict__.items():
            if isinstance(v,Config):
                ret[k] = v.restore_to_dict()
            elif isinstance(v,list):
                ret[k] = revert_list_into_list(v)
            else:
                ret[k] = v
        return ret


    def update_using_dict(self,j):
        def expand_list(root_list):
            buf = list()
            for i in root_list:
                if isinstance(i, dict):
                    buf.append(dfs_dict_binding(Config(),i))
                elif isinstance(i, list):
                    buf.append(expand_list(i))
                else:
                    buf.append(i)
            return buf

        def dfs_dict_binding(tar_dict,root):
            for k, v in root.items():
                if isinstance(v, dict):
                    tar_dict.__setattr__(k, dfs_dict_binding(Config(),v))
                elif isinstance(v, list):
                    tar_dict.__setattr__(k, expand_list(v))
                else:
                    tar_dict.__setattr__(k, v)
            return tar_dict
        dfs_dict_binding(self,j)



def ReadAndConfiguration(config_file):
    with open(config_file,'r') as fp:
        j = json.load(fp)
    c = Config()
    c.update_using_dict(j)
    c.DB_MONGOURI = "mongodb://{0}:{1}@{2}/{3}?authSource={3}".format(quote_plus(c.DB_USERNAME),quote_plus(c.DB_PASSWORD),quote_plus(c.DB_URL),quote_plus(c.DB_NAME))
    c.CONFIG_FILE_PATH = config_file
    return c

