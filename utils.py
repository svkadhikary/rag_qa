import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from logger import logging

logger = logging.getLogger("Utils_logs")

def load_pdf(file_path):
    '''
    Reads content from a pdf file and return texts as single string
    Input: file_path:str
    Output: str: Concatenated text content of all the pages in the PDF
    '''
    # read file
    reader = PdfReader(file_path)
    # loop through pages
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    logger.info(f"PDF loaded. Size: {len(text)}")

    return text

def text_splitter(text, chunk_size=100, overlap=20):
    '''
    Split text into smaller chunks
    Input: text:str, chunk_size:int, overlap:int
    Output: list of text chunks
    '''
    # create text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    # split text into chunks
    split_text = text_splitter.split_text(text)
    
    logger.info(f"Text split, chunk size {chunk_size} with overlap {overlap}")

    return split_text

def load_and_split_pdf(file_path, chunk_size=100, overlap=20):
    '''
    Load PDF and split its content into chunks
    Input: file_path:str, chunk_size:int, overlap:int
    Output: list of text chunks
    '''
    # load pdf
    text = load_pdf(file_path)
    # split text into chunks
    chunks = text_splitter(text, chunk_size, overlap)
    return chunks