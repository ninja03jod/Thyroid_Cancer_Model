from setuptools import find_packages,setup # to find the packages
from typing import List

HYPHEN_E_DOT='-e.'
def get_requirments(file_path:str)->List[str]:
    '''
    this function will return a list of requirements
    '''
    requirments = []
    with open(file_path) as file_obj:
        requirments=file_obj.readlines()
        requirments=[req.replace("\n","") for req in requirments]

        if HYPHEN_E_DOT in requirments:
            requirments.remove(HYPHEN_E_DOT)

    return requirments

setup(
    name='ThyroidCancerModel',
    author='MD.Saad',
    author_email='saadbagwan07041@gmail.com',
    packages=find_packages(),
    install_requires=get_requirments('requirments.txt')

)