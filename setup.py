from setuptools import setup, find_packages
import os


directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(name='jax-fid',
      version='0.0.2',
      url='https://github.com/matthias-wright/jax-fid',
      author='Matthias Wright',
      packages=find_packages(),
      entry_points = {
          'console_scripts': ['pycli = pycli.__main__:main']
      },
      install_requires=['numpy>=1.19.5',
                        'requests>=2.23.0',
                        'jax',
                        'jaxlib',
                        'flax',
                        'Pillow>=7.1.2',
                        'tqdm>=4.60.0',
                        'scipy'],
      extras_require={
        'testing': ['pytest'],
      },
      python_requires='>=3.6',
      license='Apache License 2.0',
      description='FID computation in Jax/Flax.',
      long_description=long_description,
      long_description_content_type='text/markdown')
