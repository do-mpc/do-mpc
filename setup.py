from setuptools import setup
from typing import List

# Read the do-mpc version
exec(open('do_mpc/_version.py').read())

# Utility to read the requirement files
def read_file_lines(file_name: str, task = 'Reading requirement files') -> List[str]:
    """ Read lines from a text file and return a list with entries.
    """
    try:
        with open(file_name, 'r') as file:
            lines = file.readlines()
            return [line.strip() for line in lines]
    except FileNotFoundError:
        print(f"Task {task} failed. File {file_name} not found.")
        return []
    except Exception as e:
        print(f"Task {task} failed. An error occurred: {str(e)}")
        return []

setup(
    name='do_mpc',
    version=__version__,
    packages=['do_mpc','do_mpc.controller','do_mpc.differentiator',
              'do_mpc.estimator','do_mpc.model','do_mpc.sampling',
              'do_mpc.sysid','do_mpc.tools', 'do_mpc.opcua',
             'do_mpc.approximateMPC'],
    author='Sergio Lucia and Felix Fiedler',
    author_email='sergio.lucia@tu-berlin.de',
    url='https://www.do-mpc.com',
    license='GNU Lesser General Public License version 3',
    long_description=open('README.md', 'r').read(),
    long_description_content_type="text/markdown",
    install_requires= read_file_lines('requirements.txt'),
    extras_require = {
        'full': read_file_lines('requirements_full.txt')[1:],
    }
)
