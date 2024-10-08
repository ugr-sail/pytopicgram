from setuptools import setup, find_packages

setup(
    name="pytopicgram",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "nltk",
        "gensim",
        "pyLDAvis",
        "tqdm",
        "argparse",
        "datetime",
    ],
    author="SAIL",
    author_email="sail@ugr.es",
    description="A Python library for Telegram crawling and topic modeling",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ugr-sail/pytopicgram",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3.10',
)
