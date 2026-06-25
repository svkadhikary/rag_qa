import os
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from logger import logging
from transformers import AutoTokenizer, AutoModel


logger = logging.getLogger("Embedding_logs")

logger.info(f"Cuda available?: {torch.cuda.is_available()}")

def transform_sentences(sentences, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    '''
    Transforms a list of sentences into their corresponding vector representations using a specified model.
    Inputs: sentences (list of str): List
    Outputs: list of vector representations
    '''
    try:
        model = SentenceTransformer(model_name,
                                    device='cuda' if torch.cuda.is_available() else 'cpu',
                                    model_kwargs={'local_files_only': True}
                                    )
        embeddings = model.encode(sentences)

        logger.info(f"Sentences encoding complete. Encoded dimensions {embeddings.shape}")
    except Exception as e:
        logger.exception("Sentence embeddings failed")
        return ""

    return embeddings

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def transformer_local(sentences):
    '''
    Transforms a list of sentences into their corresponding vector representations using a specified model.
    Inputs: sentences (list of str): List
    Outputs: list of vector representations
    '''
    sentence_transformer = "./local_minilm"
    try:
        tokenizer = AutoTokenizer.from_pretrained(sentence_transformer, local_files_only=True)
        model = AutoModel.from_pretrained(sentence_transformer, local_files_only=True)
        # Tokenize sentence
        encoded = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        # compute token embeddings
        with torch.no_grad():
            model_output = model(**encoded)
        # mean pooling
        embeddings = mean_pooling(model_output, encoded['attention_mask'])
        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)
        
        logger.info(f"Sentence Embeddings done. Embeddings shape: {str(embeddings.shape)}, Embeddings type: {str(type(embeddings))}")

    except Exception as e:
        logger.exception(f"Sentence embedding failed. {str(e)}")
        return ""
    
    return embeddings.cpu().detach().numpy()

