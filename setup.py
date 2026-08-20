from setuptools import setup, find_packages

setup(
    name="gesture_detection",
    version="0.1.0",
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
    ],
    extras_require={
        "pi": ["RPi.GPIO>=0.7.1", "luma.oled>=3.12.0", "Pillow>=9.0.0"],
        "train": ["torch>=2.0.0", "mlflow>=2.10.0"],
        "dev": ["pytest>=7.0.0"],
    },
)
