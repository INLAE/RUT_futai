from setuptools import setup, find_packages

# паспорт проекта
setup(
    name='RUT-FUT-AI',
    version='2.3.3',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'ultralytics',
        'supervision',
        'torch',
        'transformers',
        'umap-learn',
        'scikit-learn',
        'tqdm'
    ]
)