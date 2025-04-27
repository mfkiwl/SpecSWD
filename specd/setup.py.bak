from setuptools import setup, find_packages
import os 
import shutil
from distutils.command.clean import clean as Clean

class CleanCommand(Clean):
    def run(self):
        Clean.run(self)
        for path in [ 'dist', '*.egg-info']:
            if os.path.isdir(path):
                print(f"Removing {path}")
                shutil.rmtree(path)
            else:
                # Glob match for *.egg-info (in case of project_name.egg-info)
                for p in [p for p in os.listdir('.') if p.endswith('.egg-info')]:
                    print(f"Removing {p}")
                    shutil.rmtree(p)

setup(
    name="specd",
    version="0.1.0",
    author="Nanqiao Du",
    author_email="nqdu@foxmail.com",
    description="Surface wave dispersion and sensitivity analysis in anisotropic, wealy anelastic medium",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/your-repo",
    packages=find_packages(exclude=["tests", "docs"]),
    package_data={'': ['lib/*.so']},
    install_requires=[
        # List your dependencies here
        "numpy>=1.20",
        "scipy",
        "requests",
    ],
    python_requires=">=3.7",
    entry_points={
        "console_scripts": [
            "your-cli-name = your_module.cli:main",  # Adjust this path
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "License :: OSI Approved :: MIT License",
    ],
    license="MIT",
    include_package_data=True,
    zip_safe=False,

    # your existing setup() config...
    cmdclass={
        'clean': CleanCommand,
    },
)
