from tests import _TEST_ROOT, _PROJECT_ROOT, _PATH_DATA

def test_paths():
    ''' Tests the paths saved in __init__ py'''
    assert _TEST_ROOT is not None
    assert _PROJECT_ROOT is not None
    assert _PATH_DATA is not None
    
# make a function that prints hello world