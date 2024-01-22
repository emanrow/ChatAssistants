from setuptools import setup, find_packages

setup(
    name='ChatAssistants',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[  # Assuming your requirements.txt is formatted as needed
        line.strip() for line in open("requirements.txt", "r")
    ],
    test_suite='tests',
    tests_require=[
        'unittest',
    ],
    # entry_points={
    #     'console_scripts': [
    #         'chataid=ChatAssistants:main',  # If you have a main method to run
    #     ],
    # },
)
