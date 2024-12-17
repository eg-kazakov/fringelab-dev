from setuptools import setup, find_packages

setup(
    name='fringelab',
    version='0.1.2',
    description='A project for generating and visualizing fringe patterns and trajectories.',
    author='Evgeniy Kazakov',
    author_email='kazakov.eg@phystech.edu',
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'plotly',
        'dill'
    ],
    test_suite="tests",
    python_requires='>=3.6',
)
