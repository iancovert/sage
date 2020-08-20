import setuptools

setuptools.setup(
    name="sage-importance",
    version="0.0.3",
    author="Ian Covert",
    author_email="icovert@cs.washington.edu",
    description="For calculating global feature importance using Shapley values.",
    long_description="""
        SAGE (Shapley Additive Global importancE) is a game theoretic approach 
        for understanding black-box machine learning models. It summarizes each 
        feature's importance based on the predictive power it contributes, and 
        it accounts for complex interactions by using the Shapley value from 
        cooperative game theory. See the 
        [GitHub page](https://github.com/iancovert/sage/) for examples, and see 
        the [paper](https://arxiv.org/abs/2004.00668) for more details.
    """,
    long_description_content_type="text/markdown",
    url="https://github.com/iancovert/sage/",
    packages=['sage'],
    install_requires=[
        'numpy',
        'matplotlib',
        'tqdm'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.6',
)
