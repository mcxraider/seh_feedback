import pandas as pd
from groq import Groq
import json
import os
import time
import re
import sys
from tqdm import trange
from langchain_core.prompts import PromptTemplate
import logging
from math import ceil
from typing import List, Dict, Tuple

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

## load env variables
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHAT_MODEL   = os.environ["CHAT_MODEL"]
client       = Groq()

REGION = 'SG'
CSV_OUTPUT_LOCATION = f'../data/labelled_feedback/{REGION}_labelled_feedback_data_with_URL.csv'

#  load prompt from yaml file
GENERATE_EN_LABELS_PROMPT = '''
You are a linguistics professor with extensive experience in text analysis and classification. 
Your task is to first translate and then categorise seller feedback for an article webpage on an e-commerce education platform.

Follow these steps carefully:
1. **Understand the Task**: Each feedback item must be assigned one or more labels from the following list:
   - 'Negative Complaint'
   - 'Constructive Criticism'
   - 'Design Feedback'
   - 'Positive Comment'
   - 'Neutral'
   - 'Unknown'

2. **Interpretation Guidelines**:
    - Negative Complaint Expresses dissatisfaction without offering suggestions for improvement. (E.g., "The UI is terrible and frustrating to use.")
    - Constructive Criticism – Offers specific feedback on what could be improved. (E.g., "The UI could be more intuitive by reducing unnecessary steps.")
    - Design Feedback – Mentions aspects related to visual design, user experience, or layout. (E.g., "The font is too small and hard to read.")
    - Positive Comment – Expresses satisfaction or praise. (E.g., "Great platform! I love using it.")
    - Neutral – Does not express strong positive or negative sentiment. (E.g., "This feature exists.")
    - Unknown – The intent or meaning of the feedback is unclear. (E.g., "hmmm... idk.")

You are not to write any code, but just use your knowledge to classify the feedback.
Your output should be the feedback IDs and their corresponding label.

Example Output format:
[{{"feedback_id": 123456, "label": ["Negative Complaint"]}}, {{"feedback_id": 423456, "label": ["Constructive Criticism","Design Feedback"]}}, {{"feedback_id": 654321, "label": ["Negative Complaint"]}}]

Now classify the following feedback:
Feedbacks: {pairs}

Double check and ensure that your format output matches the example output format provided.
'''

