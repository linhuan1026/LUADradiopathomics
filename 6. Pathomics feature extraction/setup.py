#!/usr/bin/env python

from distutils import sysconfig
import platform
import sys

import numpy
from setuptools import Extension, setup
from setuptools.command.test import test as TestCommand
import versioneer

# Check if current PyRadiomics is compatible with current python installation (> 2.6, 64 bits)
if sys.version_info < (2, 6, 0):
    raise Exception("pypathomics requires python 2.6 or later")

if platform.architecture()[0].startswith('32'):
    raise Exception('PyRadiomics requires 64 bits python')

with open('requirements.txt', 'r') as fp:
    requirements = list(filter(bool, (line.strip() for line in fp)))

with open('requirements-dev.txt', 'r') as fp:
    dev_requirements = list(filter(bool, (line.strip() for line in fp)))

with open('requirements-setup.txt', 'r') as fp:
    setup_requirements = list(filter(bool, (line.strip() for line in fp)))

with open('README.md', 'rb') as fp:
    long_description = fp.read().decode('utf-8')


class NoseTestCommand(TestCommand):
    """Command to run unit tests using nose driver after in-place build"""

    user_options = TestCommand.user_options + [
        ("args=", None, "Arguments to pass to nose"),
    ]

    def initialize_options(self):
        self.args = []
        TestCommand.initialize_options(self)

    def finalize_options(self):
        TestCommand.finalize_options(self)
        if self.args:
            self.args = __import__('shlex').split(self.args)

    def run_tests(self):
        # Run nose ensuring that argv simulates running nosetests directly
        nose_args = ['nosetests']
        nose_args.extend(self.args)
        __import__('nose').run_exit(argv=nose_args)


commands = versioneer.get_cmdclass()
commands['test'] = NoseTestCommand

incDirs = [sysconfig.get_python_inc(), numpy.get_include()]

ext = [
    Extension("pathomics._cmatrices",
              ["pathomics/src/_cmatrices.c", "pathomics/src/cmatrices.c"],
              include_dirs=incDirs),
    Extension("pathomics._cshape",
              ["pathomics/src/_cshape.c", "pathomics/src/cshape.c"],
              include_dirs=incDirs)
]

setup(name='pypathomics',
      url='http://github.com/wuusn/pypathomics#readme',
      project_urls={
          'Radiomics.io': 'https://www.pathomics.io/',
          'Documentation':
          'https://pypathomics.readthedocs.io/en/latest/index.html',
          'Docker': 'https://hub.docker.com/r/wuusin/pypathomics/',
          'Github': 'https://github.com/wuusn/pypathomics'
      },
      author='pypathomics community',
      author_email='pypathomics@googlegroups.com',
      version=versioneer.get_version(),
      cmdclass=commands,
      packages=['pathomics', 'pathomics.scripts'],
      ext_modules=ext,
      zip_safe=False,
      package_data={
          'pathomics': ['schemas/paramSchema.yaml', 'schemas/schemaFuncs.py']
      },
      entry_points={
          'console_scripts':
          ['pypathomics=pathomics.scripts.__init__:parse_args']
      },
      description='Radiomics features library for python',
      long_description=long_description,
      long_description_content_type='text/markdown',
      license='BSD License',
      classifiers=[
          'Development Status :: 5 - Production/Stable',
          'Environment :: Console',
          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: BSD License',
          'Operating System :: OS Independent',
          'Programming Language :: C',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
          'Programming Language :: Python :: 3.4',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
      ],
      keywords='pathomics cancerimaging medicalresearch computationalimaging',
      install_requires=requirements,
      test_suite='nose.collector',
      tests_require=dev_requirements,
      setup_requires=setup_requirements)
