from setuptools import setup, find_packages

setup(
    name='EHydro_TreeUnet',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    ],
    author='Samuel Novoa',
    author_email='samuel.novoa@uvigo.gal',
    description='Un modelo tipo U-Net con cabeza para segmentación de instancia para nubes de puntos dispersas.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GeoTechUVigo/EHydro_TreeUnet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Ubuntu 20.04',
    ],
    python_requires='>=3.6',
)