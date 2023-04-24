from setuptools import setup

exec(open('do_mpc/_version.py').read())

setup(
    name='do_mpc',
    version=__version__,
    packages=['do_mpc','do_mpc.controller','do_mpc.differentiator',
              'do_mpc.estimator','do_mpc.model','do_mpc.sampling',
              'do_mpc.sysid','do_mpc.tools'],
    author='Sergio Lucia and Felix Fiedler',
    author_email='sergio.lucia@tu-berlin.de',
    url='https://www.do-mpc.com',
    license='GNU Lesser General Public License version 3',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    install_requires=[
        "casadi",
        "numpy",
        "matplotlib"
    ],
)
