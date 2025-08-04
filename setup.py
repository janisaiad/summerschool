from setuptools import setup
import os


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ["PROJECT_ROOT"] = PROJECT_ROOT

if __name__ == "__main__":
    setup()