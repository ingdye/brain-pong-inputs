from setuptools import setup, find_packages

setup(
    name="brain_pong_inputs",
    version="0.0.1",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    license="MIT",
    author="Dan Gale",
    long_description=open("README.md").read(),
    install_requires=[],
    tests_require=["pytest", "pytest-cov"],
    setup_requires=["pytest-runner"],
    package_data={"my_package": ["data.json"]},
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "brain_pong_inputs=brain_pong_inputs.main:main",
        ],
    },
)
