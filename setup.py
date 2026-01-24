from setuptools import setup, find_packages

setup(
    name="lahteam_tuner",
    version="1.0.0",
    author="LahTeam.VN",
    author_email="contact@lahteam.vn",
    description="Helper library for Musubi Tuner training on Google Colab",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/LahTeam/lahteam_tuner",
    packages=["lahteam_tuner"],
    package_dir={"lahteam_tuner": "."},
    py_modules=["config", "download", "utils"],
    python_requires=">=3.8",
    install_requires=[
        "toml",
        "huggingface_hub",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
