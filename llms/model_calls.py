import os
import numpy as np
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from logger import logging
from PydanticSchemas.schemas import EvaluationDataset

logger = logging.getLogger("ChatLLM_calls_log")

with open("SECRETS.txt", "r") as secr:
    groq_api_key = secr.readlines()[0].split("=")[-1].strip()

class GroqChatModel:
    def __init__(self, model_name="llama-3.3-70b-versatile", temperature=0.2, max_tokens=1024):
        
        # self.model_name = model_name
        # self.temperature = temperature
        # self.max_tokens = max_tokens
        try:
            self.client = ChatGroq(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                api_key=groq_api_key
            )

            logger.info(f"LLM initialized: {model_name}, temperature: {temperature}")
        except Exception as e:
            logger.exception("Encountered an error while connecting to Groq")
    
    def format_messages(self, query, context=None):
        
        prompt = f"""You are a helpful assistant. Use the following context to answer the user's question accurately. 
            If the answer cannot be found in the context, politely state that you do not know.

            Context:
            {context}

            Question: {query}
            Answer:"""
        
        logger.info(f"Formatted prompt with contexts")
        
        return prompt
    
    def generate_responses(self, query, context=None):
        # format the prompt with query and context
        prompt = self.format_messages(query, context)
        try:
            # generate response from the model
            response = self.client.invoke(prompt)
            # print("Raw model response:", response)
            logger.info(f"Response content: {response.content}")
            logger.info(f"Response Metadata: {response.response_metadata}")
            return response.content
        
        except Exception as e:
            logger.exception("Could not get a response from LLM")
            return ""

    
    def query_rewrite(self, query):
        # enhance user query
        prompt = f"You are a helpful assistant. You are skilled in rephrasing sentences to enrich its contents \
            which optimizes it for using in retrieval related tasks, such as RAG.\
                Rephrase and optimize the sentence inside the quotes: '{query}'\
                    Make sure you do not put irrelevant context in response."
        # generate response
        try:
            response = self.client.invoke(prompt)
            logger.info(f"User query: {query}")
            logger.info(f"Enriched query: {response.content}")
            return response.content
        
        except Exception as e:
            logger.exception("Could not get a response from LLM")
            return ""
    
    def create_test_qa(self, text):
        '''
            Creates question and ground truth to evaluate RAG
            input: text (data) to create question and ground truth from
            output:
        '''
        # initialize json parser with evaluation datset schema
        parser = JsonOutputParser(pydantic_object=EvaluationDataset)
        # construct Prompt template
        prompt = PromptTemplate(
            template="You are an expert AI evaluator." \
            "Based on the source text provided below, generate exactly 5 to 6 distinct evaluation question and ground truth answer pairs." \
            "\n\n{format_instructions}\n\nSource Text:\n{context}",
            input_variables=['context'],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        # assemble the chain
        chain = prompt | self.client | parser

        # run chain to get clean python dictionary
        try:
            extracted_data = chain.invoke({"context": text})
            logger.info(f"Generating clean dictionary of question and ground truths")
        except Exception as e:
            logger.exception(f"Could not generate: {e.__traceback__}")
            return None
        
        return extracted_data



