from setuptools import setup, find_packages

setup(
    name="rag_wiki",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "datasets",
        "transformers",
        "torch",
        "numpy",
        "faiss",
        "faiss-gpu",
    ],
    description="A Python package for RAG from Wikipedia.",
    author="Shweta Agrawal",
    url="https://github.com/ashweta1/rag_wiki",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)