from setuptools import find_packages, setup

with open("requirements.txt", "r") as f:
    requirements = f.read().splitlines()

setup(
    name='huggingpics',
    packages=find_packages(exclude=['examples']),
    version='0.0.1',
    license='MIT',
    description='🤗🖼️ HuggingPics: Fine-tune Vision Transformers for anything using images found on the web.',
    author='Nathan Raw',
    author_email='naterawdata@gmail.com',
    url='https://github.com/nateraw/huggingpics',
    install_requires=requirements,
)