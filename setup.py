	from setuptools import setup

setup(name='cellfate',
      version='0.1',
      description='Fitting parameters of the model for density \
                  of human embryonic stem cells in processed images at different \
                  time t to differentiate the fates of the cells',
      url='https://github.com/p201-sp2016/cellfate',
      author='Tessa Green, Yau Chuen (Oliver) Yam, Seung Hwan Lee',
      author_email='tessa_green@g.harvard.edu,\
                  yyam@g.harvard.edu,\
                  lee_seunghwan@g.harvard.edu',
      license='GNU GPL v3',
      packages=['cellfate'],
      # All dependencies should be here.
      # If dependencies are not on PyPI, use URL. See:
      # https://python-packaging.readthedocs.org/en/latest/dependencies.html
      install_requires=[
          'numpy',
          'scipy',
          'emcee',
          'seaborn',
          'matplotlib',
          'pandas'
      ],
      zip_safe=False,
      test_suite='nose.collector',
      tests_require=['nose'],)
