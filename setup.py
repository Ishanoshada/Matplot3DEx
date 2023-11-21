from setuptools import setup, find_packages


with open('README.md', 'r') as f:
    long_description = f.read()


setup(
    name='Matplot3DEx',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'seaborn'
    ],
    author='Ishan Oshada',
    author_email='ishan.kodithuwakku@gmail.com',
    description='A Matplotlib 3D Extension package for enhanced data visualization',
   long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/ishanoshada/matplot3dex',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)
