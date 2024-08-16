from setuptools import setup

setup(
    name='TransformCamK',
    version='0.0.1',
    description='Camera intrinsic processor with image processing',
    author='Geonung Kim',
    author_email='saywooong@gmail.com',
    packages=[
        'TransCamK',
    ],  # same as name
    # install_requires=['wheel', 'bar', 'greek'], #external packages as dependencies
    # entry_points={
    #     'console_scripts': [
    #         'pycomar = pycomar.io:hello',
    #     ],
    # },
)
