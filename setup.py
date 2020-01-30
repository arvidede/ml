from setuptools import setup, find_packages

setup(
   name='ml',
   version='0.1',
   description='',
   author='Arvid Edenheim',
   include_package_data=True,
   author_email='',
   packages=find_packages(),  #same as name
   install_requires=['numpy'], #external packages as dependencies
)
