import io
import os
import sys

from setuptools import setup, find_packages


def get_version():
    return '0.0.1'


def get_test_requirements():
    requirements = ['pytest']
    if sys.version_info < (3, 3):
        requirements.append('mock')
    return requirements


def get_long_description():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with io.open(os.path.join(base_dir, 'README.md'), encoding='utf-8') as f:
        return f.read()


setup(
    name='helper_functions',
    version=get_version(),
    description='Unstructured helper functions',
    long_description=get_long_description(),
    long_description_content_type='text/markdown',
    author='Vladimir Iglovikov',
    license='MIT',
    url='',
    packages=find_packages(exclude=['tests']),
    install_requires=['numpy',
                      'pandas'],
    extras_require={'tests': get_test_requirements()},
    classifiers=[
        'Development Status :: 4 - Beta',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.0',
        'Programming Language :: Python :: 3.1',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
)
