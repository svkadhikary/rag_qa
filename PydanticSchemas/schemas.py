import os
from typing import List
from pydantic import BaseModel, Field

class QAPair(BaseModel):
    
    question: str = Field(description="A spcific question based strictly on the provided context.")
    ground_truth: str = Field(description="The exact factual answer sourced directly from the context.")

class EvaluationDataset(BaseModel):

    dataset: List[QAPair] = Field(description="A list of 5 to 6 distinct question and ground truth pairs.")
    