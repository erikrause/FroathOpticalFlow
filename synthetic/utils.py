import yaml


def get_config(path):
    fconfig = open(path)
    config = yaml.safe_load(fconfig)
    fconfig.close()
    return config
