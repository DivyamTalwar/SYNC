from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="sync-system",
    version="0.1.0",
    description="Multi-agent LLM collaboration system with cognitive synergy",
    author="SYNC Development Team",
    author_email="dev@sync-system.ai",
    url="https://github.com/divyamtalwar/sync",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="multi-agent llm reinforcement-learning collaboration ai",
    entry_points={
        "console_scripts": [
            "sync-pretrain=scripts.pretrain_ckm:main",
            "sync-train=scripts.train_rl:main",
            "sync-evaluate=scripts.evaluate:main",
            "sync-demo=scripts.demo:main",
        ],
    },
)
