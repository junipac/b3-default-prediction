from setuptools import setup, find_packages

setup(
    name="b3_ingestion",
    version="1.0.0",
    description="Módulo de ingestão resiliente de dados B3/CVM para modelagem de PD",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "requests>=2.31.0",
        "pandas>=2.2.0",
        "pyarrow>=16.0.0",
        "sqlalchemy>=2.0.0",
        "psycopg2-binary>=2.9.0",
        "tenacity>=8.3.0",
        "structlog>=24.0.0",
        "python-dotenv>=1.0.0",
        "click>=8.1.0",
        "rich>=13.7.0",
        "numpy>=1.26.0",
        "aiohttp>=3.9.0",
    ],
    entry_points={
        "console_scripts": [
            "b3-ingest=main:cli",
        ],
    },
)
