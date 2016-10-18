from setuptools import setup

setup(name='tfr',
      version='0.1',
      description='Time-frequency reassigned spectrograms',
      url='http://github.com/bzamecnik/tfr',
      author='Bohumir Zamecnik',
      author_email='bohumir.zamecnik@gmail.com',
      license='MIT',
      packages=['tfr'],
      zip_safe=False,
      install_requires=[
         'numpy',
         'scikit-learn',
         'scipy',
      ],
      # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Intended Audience :: Science/Research',
          'Topic :: Multimedia :: Sound/Audio :: Analysis',

          'License :: OSI Approved :: MIT License',

          'Programming Language :: Python :: 3',

          'Operating System :: POSIX :: Linux',
          'Operating System :: MacOS :: MacOS X',
      ])
