import pytest





def test_env():
    try:
        import sciml
    except ImportError:
        pytest.fail("sciml is not installed")
    
