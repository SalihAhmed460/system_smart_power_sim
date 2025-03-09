from setuptools import setup, find_packages

setup(
    name="tensorflow-compat",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib"
    ],
    description="TensorFlow compatibility layer using scikit-learn",
    author="User",
)
