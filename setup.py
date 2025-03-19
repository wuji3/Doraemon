from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='doraemon-torch', 
    version='0.0.3alpha',
    author='duke',
    author_email='dk812821001@163.com',
    description='Doraemon',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/wuji3/Doraemon',
    packages=find_packages(include=['doraemon', 'doraemon.*']),
    python_requires='>=3.10',
    install_requires=[
        'torchmetrics==0.11.4',
        'opencv-python==4.7.0.72',
        'numpy==1.24.3',
        'tqdm==4.66.4',
        'Pillow==9.4.0',
        'grad-cam==1.4.8',
        'timm==0.9.16',
        'tensorboard==2.16.2',
        'prettytable==3.10.0',
        'datasets==2.20.0',
        'imagehash==4.3.1',
        'transformers==4.48.3',
        'torch==2.5.1',
        'torchvision==0.20.1',
        'torchaudio==2.5.1',
        'faiss-cpu==1.7.2',
    ],
)
