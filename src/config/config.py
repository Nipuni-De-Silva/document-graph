import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    NEO4J_URI = os.getenv("NEO4J_URI","neo4j+s://c95a3680.databases.neo4j.io")
    NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "5SYecqiUcLZz4pzO9CDdGs9jlU5rOKUQ6ddtK6DEl1o")

    
    DEEPINFRA_API_TOKEN = os.getenv("DEEPINFRA_API_TOKEN", "YuGM4YMWqQU4kVM0u47Ntev9gUjFv2Om")
