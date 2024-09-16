from setuptools import setup, find_packages

setup(
    name="apps_metric",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "evaluate",
        "datasets",
        # 'typing',
        # 'multiprocessing',
        # 'platform',
        # 'datetime',
        # 'signal',
        # 'numpy',
        # 'unittest',
        "pyext",
        # 'pprint',
        # 'tqdm',
        # 'transformers',
    ],
)
