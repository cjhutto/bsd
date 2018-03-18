import pip
from setuptools import setup, find_packages
from setuptools.command.install import install


package = 'bsdetector'
version = '0.1'
links = []  # for repo urls (dependency_links)
requires = []  # for package names

# new versions of pip requires a session
requirements = pip.req.parse_requirements(
    'requirements.txt', session=pip.download.PipSession()
)

for item in requirements:
    link = None
    if getattr(item, 'url', None):  # older pip has url
        link = str(item.url)
    if getattr(item, 'link', None):  # newer pip has link
        link = str(item.link)
    if link:
        link = link.lstrip('git+')
        links.append(link)
    if item.req:
        item_req = str(item.req)
        if item_req == 'pattern':
            continue
        requires.append(item_req)  # always the package name


class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        # PUT YOUR POST-INSTALL SCRIPT HERE or CALL A FUNCTION
        import nltk
        nltk.download('punkt')
        install.run(self)


setup(
    name=package,
    version=version,
    package_dir={'bsdetector': 'bsdetector'},
    package_data={'bsdetector': ['*.json']},
    packages=find_packages(),
    install_requires=requires,
    dependency_links=links,
    include_package_data=True,
    description="Detects biased statements in online media documents",
    url='url',
    cmdclass={
        'install': PostInstallCommand,
    }
)
