# setup.py
from setuptools import setup
import subprocess
import os

BASE_PACKAGE_NAME = 'dafne-models'

REQUIREMENTS = {
    '*': [
        'matplotlib',
        'ormir-pyvoxel',
        'dafne-dicomUtils >= 0.1.7',
        'dill'
    ],
    'main': [
        'dafne_dl >= 1.4a2'
    ],
    'torch_models': [
        'dafne_dl-torch_models >= 2.0a2'
    ]
}


def get_current_branch():
    """Get the current git branch name."""
    # First try environment variable (for CI/CD environments)
    branch = os.getenv('GIT_BRANCH')
    if branch:
        return branch.split('/')[-1]  # Handle cases like 'origin/main'

    # Then try git command
    try:
        branch = subprocess.check_output(
            ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
            stderr=subprocess.DEVNULL  # Suppress stderr
        ).decode('utf-8').strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'main'  # Default fallback


def get_package_name():
    """Determine package name based on git branch."""
    branch = get_current_branch()

    if branch == 'main':
        return BASE_PACKAGE_NAME
    else:
        return f'{BASE_PACKAGE_NAME}-{branch}'

def get_requirements():
    """Get the requirements for the current branch."""
    branch = get_current_branch()
    requirements_common = REQUIREMENTS.get('*', [])
    reqs = requirements_common + REQUIREMENTS.get(branch, REQUIREMENTS['main'])
    print(reqs)
    return reqs

setup(
    name=get_package_name(),
    install_requires=get_requirements()
    # Other setup parameters will be read from setup.cfg
)