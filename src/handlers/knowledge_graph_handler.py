import os
import hashlib
import pickle
import spacy
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
# from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.llms import DeepInfra
from src.config.config import Config
from neo4j import GraphDatabase
from langchain_experimental.graph_transformers import LLMGraphTransformer

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_community.graphs import Neo4jGraph
from langchain_community.document_loaders import TextLoader
from yfiles_jupyter_graphs import GraphWidget


class KnowledgeGraphHandler:
    def __init__(self):
        # Initialize Neo4j driver with credentials from Config
        self.driver = GraphDatabase.driver(
            Config.NEO4J_URI,
            auth=(Config.NEO4J_USERNAME, Config.NEO4J_PASSWORD)
        )
        # Load the SpaCy NLP model for named entity recognition
        self.nlp = spacy.load("en_core_web_sm")
        # Load the DeepInfra API token for the LLM
        self.deepinfra_api_token = Config.DEEPINFRA_API_TOKEN

    # Generate a unique hash for the file based on its name and size
    def _get_filename_hash(self, file_path):
        filename = os.path.basename(file_path)    # Extract the file name
        file_size = os.path.getsize(file_path)    # Get the file size
        hasher = hashlib.md5(f"{filename}_{file_size}".encode())    # Combine and hash
        return hasher.hexdigest()

    # Extract entities from text using SpaCy's NER
    def _extract_entities(self, text):
        doc = self.nlp(text)
        return [(ent.text, ent.label_) for ent in doc.ents]

    # Process the documents and generate graph representations
    def _process_documents(self, file_path):
        # Generate a hash for the file to uniquely identify its cache
        file_hash = self._get_filename_hash(file_path)
        cache_file = f"./cache/graph_documents_{file_hash}.pkl"

        if os.path.exists(cache_file):
            # Load cached graph documents if they exist
            with open(cache_file, "rb") as f:
                graph_documents = pickle.load(f)
            print(f"Loaded graph_documents from cache for {file_path}.")
        else:
            pdf_loader = PyMuPDFLoader(file_path=file_path)
            docs = pdf_loader.load()

            # Split the document into smaller chunks for processing
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            documents = text_splitter.split_documents(documents=docs)


            # Initialize the DeepInfra LLM with specified model and parameters
            llm = DeepInfra(
                model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
                deepinfra_api_token=self.deepinfra_api_token
            )
            llm.model_kwargs = {"temperature": 0}
            llm_transformer = LLMGraphTransformer(llm=llm)

            # Convert the document chunks into graph documents
            graph_documents = llm_transformer.convert_to_graph_documents(documents)


            # Save the generated graph documents to a cache file
            with open(cache_file, "wb") as f:
                pickle.dump(graph_documents, f)
            print(f"Generated and saved graph_documents to cache for {file_path}.")
        
        return graph_documents

    # Save the graph documents into the Neo4j knowledge graph
    def _save_to_graph(self, graph_documents):
        with self.driver.session() as session:
            for graph_document in graph_documents:
                for node in graph_document.nodes:
                    session.run(
                        """
                        MERGE (n:`{type}` {{id: $id}})
                        SET n += $properties
                        SET n:__Entity__
                        """.format(type=node.type),
                        id=node.id, properties=node.properties
                    )

                for relationship in graph_document.relationships:
                    session.run(
                        """
                        MATCH (a {{id: $source_id}}), (b {{id: $target_id}})
                        MERGE (a)-[r:{type}]->(b)
                        SET r += $properties
                        """.format(type=relationship.type),
                        source_id=relationship.source.id,
                        target_id=relationship.target.id,
                        properties=relationship.properties
                    )

    # Orchestrates the processing of a document and saving to the graph
    def process_and_save_document(self, file_path):
        graph_documents = self._process_documents(file_path)
        self._save_to_graph(graph_documents)
        print(f"Data from {file_path} has been processed and saved to the knowledge graph.")