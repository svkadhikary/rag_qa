import os
import cohere
from logger import logging

logger = logging.getLogger("Reranker_logs")

with open("SECRETS.txt", 'r') as secr:
    co_api_key = secr.readlines()[-1].split("=")[-1].strip()

try:
    co = cohere.ClientV2(api_key=co_api_key)
    logger.info(f"Cohere API intitialized!")
except Exception as e:
    logger.exception("Cannot connect to Cohere")

def cohere_rerank(query, docs, top_k=5):

    if not query or not docs:
        return ""
    
    response = ""

    if isinstance(docs, list):

        try:
            response = co.rerank(
                model="rerank-v4.0-pro",
                query=query,
                documents=docs,
                top_n=top_k,
            )
        
            logger.info(f"Reranking success: \n{response.results}")
            
        except Exception as e:
            logger.exception("No response from LLM")
            return ""

    top_docs = [docs[result.index] for result in response.results]
    
    return top_docs
