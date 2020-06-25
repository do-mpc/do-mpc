from setuptools import setup

setup(
    name='do_mpc',
    version='4.0.0',
    packages=['do_mpc','do_mpc.tools'],
    author='Sergio Lucia and Felix Fiedler',
    author_email='sergio.lucia@tu-berlin.de',
    url='https://www.do-mpc.com',
    license='GNU LESSER GENERAL PUBLIC LICENSE',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "casadi",
        "numpy",
        "matplotlib"
    ],
)
