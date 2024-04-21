import os
from llama_index.core.text_splitter import TokenTextSplitter, SentenceSplitter
from llama_index.core.schema import Document
from llama_index.core.readers.download import download_loader
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
    BaseExtractor,
)
from llama_index.core.schema import MetadataMode
from llama_index.core.ingestion import IngestionPipeline

from modules.loaders import LOADERS
import yake
import re
import time

from app.config import EMBEDDINGS_PATH

def build_pipeline(llm):

    transformations = [
        SentenceWindowNodeParser.from_defaults(
            window_size=3,
            window_metadata_key="window",
            original_text_metadata_key="original_text",
        ),
        KeywordExtractor(
            llm=llm, keywords=10
        ),
        SummaryExtractor(
            llm=llm, metadata_mode=MetadataMode.EMBED, num_workers=8
        ),
    ]

    return IngestionPipeline(transformations=transformations)

def findVectorDB(project):
    if project.model.vectorstore == "redis":
        from app.vectordb.redis import RedisVector
        return RedisVector
    elif project.model.vectorstore == "chroma":
        from app.vectordb.chromadb import ChromaDBVector
        return ChromaDBVector
    elif project.model.vectorstore == "pinecone":
        from app.vectordb.pinecone import PineconeVector
        return PineconeVector
    else:
        raise Exception("Invalid vectorDB type.")


def IndexDocuments(project, documents, splitter="sentence", chunks=256, llm=None):
    if splitter == "sentence":
        splitter_o = TokenTextSplitter(
            separator=" ", chunk_size=chunks, chunk_overlap=30)
    elif splitter == "token":
        splitter_o = SentenceSplitter(
            separator=" ", paragraph_separator="\n", chunk_size=chunks, chunk_overlap=30)
    elif splitter == "pipeline":
        
        pipeline = build_pipeline(llm)


        nodes = pipeline.run(documents)
        project.vector.index.insert_nodes(nodes)
        return len(nodes)


    for document in documents:
        text_chunks = splitter_o.split_text(document.text)

        doc_chunks = [Document(text=t, metadata=document.metadata)
                      for t in text_chunks]

        for doc_chunk in doc_chunks:
            project.vector.index.insert(doc_chunk)

    return len(doc_chunks)


def ExtractKeywordsForMetadata(documents):
    
    max_ngram_size = 4
    numOfKeywords = 15
    kw_extractor = yake.KeywordExtractor(n=max_ngram_size, top=numOfKeywords)
    for document in documents:
        metadataKeywords = ""
        keywords = kw_extractor.extract_keywords(document.text)
        for kw in keywords:
            metadataKeywords = metadataKeywords + kw[0] + ", "
        document.metadata["keywords"] = metadataKeywords

    return documents


def FindFileLoader(ext, eargs={}):
    if ext in LOADERS:
        loader_name, loader_args = LOADERS[ext]
        loader = download_loader(loader_name)()
        return loader
    else:
        raise Exception("Invalid file type.")


def FindEmbeddingsPath(projectName):
    embeddings_path = EMBEDDINGS_PATH
    embeddingsPathProject = None

    if not os.path.exists(embeddings_path):
        os.makedirs(embeddings_path)

    project_dirs = [d for d in os.listdir(
        embeddings_path) if os.path.isdir(os.path.join(embeddings_path, d))]

    for dir in project_dirs:
        if re.match(f'^{projectName}_[0-9]+$', dir):
            embeddingsPathProject = os.path.join(embeddings_path, dir)

    if embeddingsPathProject is None:
        embeddingsPathProject = os.path.join(
            embeddings_path, projectName + "_" + str(int(time.time())))
        os.mkdir(embeddingsPathProject)

    return embeddingsPathProject


