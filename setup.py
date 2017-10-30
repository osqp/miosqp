from setuptools import setup

setup(
    name='miosqp',
    version='0.0.1',
    author='Bartolomeo Stellato, Vihangkumar V. Naik',
    author_email='bartolomeo.stellato@gmail.com, vihangkumar.naik@imtlucca.it',
    packages=['miosqp'],
    package_dir={'miosqp': 'miosqp'},
    url='http://github.com/oxfordcontrol/miosqp',
    license='Apache v2.0',
    description='An MIQP solver based on OSQP',
    install_requires=["osqp",
                      "numpy >= 1.9",
                      "scipy >= 0.15"]
)
