import pytest




def test_deeponet():
    try:
        from sciml.model.deeponet import DeepONet
    except ImportError:
        pytest.fail("DeepONet is not installed")
    
    
    