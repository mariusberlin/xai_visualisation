from setuptools import setup

setup(
 
    name='XAI visualisation',
    url='https://github.com/mariusberlin/xai_visualisation/xai_vis',
    author='Marius Pullig',
    author_email='pullig.marius@gmail.com',
    # Needed to actually package something
    packages=['xai_vis'],
    # Needed for dependencies
    install_requires=['numpy'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='3D visualisation of interpretability methods in Jupyter Notebook',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
    )
