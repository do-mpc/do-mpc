from setuptools import setup

setup(
    name='do_mpc',
    version='4.0.0dev',
    packages=['do_mpc','do_mpc.tools'],
    author='Sergio Lucia',
    author_email='sergio.lucia@tu-berlin.de',
    license='LICENSE.txt',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "casadi >= 3.5.1",
        "numpy >= 1.17.2",
        "jupyter >= 1.0.0",
        "matplotlib >= 3.1.1"
    ],
)
