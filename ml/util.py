REGISTRY = {}


def register(type_of, name):
    """
    Register a class with the registry
    :param type_of: the top level type of the class. it will be the first key in the registry
    :param name: the name of the class. it will be the second key in the registry
    :return: the class
    """

    def inner(cls):
        if type_of not in REGISTRY:
            REGISTRY[type_of] = {}
        REGISTRY[type_of][name] = cls
        return cls

    return inner


def get_class(type_of, name):
    """
    Get a class from the registry
    :param type_of: the top level type of the class
    :param name: the name of the class
    :return: the class
    """
    if type_of not in REGISTRY:
        raise ValueError(f'Unknown type {type_of}')
    if name not in REGISTRY[type_of]:
        raise ValueError(f'Unknown {type_of} {name}')

    return REGISTRY[type_of][name]
