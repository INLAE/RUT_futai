from setuptools import setup, find_packages

setup(
    name='team_video_tracker',
    version='0.1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
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