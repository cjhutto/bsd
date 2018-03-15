from setuptools import setup, find_packages

package = 'bsdetector'
version = '0.1'
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install


# class PostDevelopCommand(develop):
#     """Post-installation for development mode."""
#     def run(self):
#         # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
#         develop.run(self)

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        import nltk
        nltk.download('punkt')
        install.run(self)

print(find_packages('bsdetector'))
setup(name=package,
      version=version,
      packages=['bsdetector', 'additional_resources'],
      install_requires=['decorator', 'requests', 'textstat', 'vaderSentiment',
                        'pattern', 'nltk', 'pytest'],
      package_dir={'bsdetector': 'bsdetector'},
      # data_files=[('bsdetector', ['bsdetector/lexicon.json'])],
      package_data={'bsdetector': ['*.json']},
      description="Detects biased statements in online media documents",
      url='url',
      cmdclass={
          # 'develop': PostDevelopCommand,
          'install': PostInstallCommand,
      }
)

