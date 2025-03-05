import pytest

def test_arbitrary(): 
    assert 1 == 1
    
def test_fc_consistency(): 
    """Test whether calc_fc is outputting a consistent matrix. 
    
    """
    
    #read in exemplar matrix
    
    #compare expemplar matrix to calculated matrixs