from setuptools import setup, find_packages


setup(
    name="flaq",
    version="0.0.1",
    description="Quantum flag-based codes",
    license='MIT',
    packages=find_packages(),
    author='Arthur Pesah',
    email='arthur.pesah.20@ucl.ac.uk',
    setup_requires=['setuptools_scm'],
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'ldpc', 'networkx', 'pyvis'],
)
