from setuptools import setup, find_packages

setup(
    name='Panoramix3D',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
    ],
    author='Samuel Novoa',
    author_email='samuel.novoa@uvigo.gal',
    description='Panoptic Object Recognition And voxel-to-centroid Correlation in 3D.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/GeoTechUVigo/EHydro_TreeUnet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: Ubuntu 20.04',
    ],
    python_requires='>=3.6',
)