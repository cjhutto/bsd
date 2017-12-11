from setuptools import setup, find_packages

package = 'bsdetector'
version = '0.1'

print(find_packages('bsdetector'))
setup(name=package,
      version=version,
      packages=['bsdetector', 'ref_lexicons', 'additional_resources'],
      install_requires=['decorator', 'requests', 'textstat', 'vaderSentiment', 'pattern', 'nltk', 'pytest'],
      description="Detects biased statements in online media documents",
      url='url')
