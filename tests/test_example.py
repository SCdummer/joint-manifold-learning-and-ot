"""This is an example of how you would write a test. Typically the tested function, 
in this case 'addition', is imported from another file. 
"""

def addition(x, y):
    """Adds the input parameters

    Keyword arguments:
    x -- base number
    y -- number to be added to x
    """
    return  x + y


def test_addition_positive():
    """Test addition function for two positive numbers"""
    x = 2
    y = 3
    z = addition(x,y)
    assert ( z == 5 ), "Some useful error message"
