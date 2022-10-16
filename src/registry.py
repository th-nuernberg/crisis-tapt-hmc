import collections

_REGISTRY = collections.defaultdict(dict)


def register(name):
    registry = _REGISTRY
    def decorator(obj):
        if name in registry:
            raise LookupError("{} already registered".format(name))
        registry[name] = obj
        return obj
    return decorator


def lookup(name):
    if not name in _REGISTRY:
        raise KeyError("{} not registered".format(name))
    return _REGISTRY[name]


def make(config, **kwargs):
    obj = lookup(name=config["name"])
    params = {**config["params"], **kwargs}
    return obj(**params)