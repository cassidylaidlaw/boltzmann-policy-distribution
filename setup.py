from distutils.core import setup
setup(
    name='boltzmann-policy-distribution',
    packages=[
        'bpd',
        'bpd.agents',
        'bpd.envs',
        'bpd.models',
    ],
    package_data={'bpd': ['py.typed']},
    version='0.0.3',
    license='MIT',
    description='Code for the ICLR 2022 paper "The Boltzmann Policy Distribution: Accounting for Systematic Suboptimality in Human Models"',
    author='Cassidy Laidlaw',
    author_email='cassidy_laidlaw@berkeley.edu',
    url='https://github.com/cassidylaidlaw/boltzmann-policy-distribution',
    download_url='https://github.com/cassidylaidlaw/boltzmann-policy-distribution/archive/TODO.tar.gz',
    keywords=['human-robot interaction', 'machine learning', 'reinforcement learning'],
    install_requires=[
        'torch>=1.9.0',
        'ray[rllib]>=1.11.0',
        'matplotlib>=3.4.1',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
) 
