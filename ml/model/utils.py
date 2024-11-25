from ml.util import register


def register_model(cls=None, *, dataset=None, name=None):
    """
    A decorator for registering model classes.
    """

    def _register(model_class):
        return register('model', f'{dataset}-{name}')(model_class)

    if cls is None:
        return _register
    return _register(cls)
