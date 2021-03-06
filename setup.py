import setuptools

with open("README.md", "r", encoding="utf8") as fh:
    long_description = fh.read()

requirements = []

setuptools.setup(
    name="tism",
    version="1.0.5",
    author="Cyril Meyer",
    author_email="contact@cyrilmeyer.eu",
    description="TensorFlow Image Segmentation Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Cyril-Meyer/tism",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    python_requires='>=3.6',
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)
