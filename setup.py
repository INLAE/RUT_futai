from setuptools import setup, find_packages

setup(
    name='RUT-FUT-AI',
    version='2.1.0',
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
    ],
    entry_points={
        'console_scripts': [
            'tvt-demo = team_video_tracker.processor:main'
        ]
    }
)