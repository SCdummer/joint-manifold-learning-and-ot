from .ot_regulariser import FastConvolutionalW2Cost


def get_regulariser(name, **kwargs):
    if name == 'ot':
        return FastConvolutionalW2Cost(**kwargs)
    else:
        raise ValueError(f'Unknown regulariser {name}')


__all__ = [
    'get_regulariser'
]
