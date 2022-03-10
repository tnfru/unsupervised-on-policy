import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="active-pre-train-ppg",
    version="0.0.4",
    author="Lars Mueller",
    author_email="lamue120@hhu.de",
    description="Unsupervised pre-training with PPG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tnfru/unsupervised-on-policy",
    project_urls={
        "Bug Tracker": "https://github.com/tnfru/unsupervised-on-policy/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'torch',
        'matplotlib',
        'wandb',
        'ale-py',
        'gym[atari, accept-rom-license]>=0.21.0',
        'kornia',
        'supersuit',
        'stable_baselines3',
        'einops'
    ],
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)

