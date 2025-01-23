import pandas as pd
from groq import Groq
import json
import os
import re
from langchain_core.prompts import PromptTemplate
import random
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

## load evv variables
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHAT_MODEL = os.environ["CHAT_MODEL"]
client = Groq()

GENERATE_LABELS_PROMPT = '''
You are an linguistics professor tasked with classifying seller feedback for an e-commerce platform. 
Each feedback item should be categorised into one or more appropriate labels from the following list:
['Negative_Complaint','Constructive_Criticism','Design_Feedback','Positive Comment','Neutral']
You are not to write any code, but just use your knowledge to classify the feedback.
Now classify the following feedback:
Feedback: {text}

Example Output format:
{{
    "feedback": "This platform is very confusing to use. The design needs improvement.",
    "label": "Negative_Complaint"
}}
''' 

def extract_answer(input_string):
    # Find the start and end indices of the JSON data within the input string
    # Assuming the JSON data starts with '{' and ends with '}'
    json_start = input_string.find("{")
    json_end = input_string.rfind("}") + 1

    # If either the start or end index is not found, raise an error
    if json_start == -1 or json_end == -1:
        raise ValueError("Invalid input: No JSON data found.")

    # Extract the substring that potentially contains the JSON data
    json_data = input_string[json_start:json_end]

    try:
        # Attempt to convert the JSON string to a Python dictionary
        data_dict = json.loads(json_data)
        return data_dict

    except json.JSONDecodeError:
        # If JSON decoding fails, search for a JSON object containing the 'questions' key
        # Using regex to match a pattern that includes the 'questions' key
        pattern = r'\{\s*"feedback":\s*".*?",\s*"label":\s*".*?"\s*\}'
        match = re.search(pattern, input_string, re.DOTALL)

        if match:
            # If a match is found, extract the matched JSON string and convert it to a dictionary
            data_json_str = match.group(0)
            data_dict = json.loads(data_json_str)
            return data_dict

        # If no valid JSON is found, the function will Log an error
        else:
            logging.error(
                "No dictionary with 'questions' as a key found in this input string. Error by LLM"
            )
            return {"error": "No dictionary with questions found"}


def generate_labels(feedback, label_prompt, client):
    prompt = PromptTemplate(
        template=label_prompt,
        input_variables=["text"],
    )

    final_prompt = prompt.format(text=feedback)

    # Generate the completion by interacting with the language model API
    completion = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
                    {
                        "role": "user", 
                        "content": final_prompt
                    }
                    ],
        temperature=0,  # Control the randomness of the output (lower means less random)
        max_tokens=1024,  # Limit the response length
        top_p=1,  # Nucleus sampling parameter (1 means only the most likely tokens are considered)
        stream=True,  # Enable streaming of the response chunks
        stop=None,  # Define stopping conditions (None means no stopping condition)
    )

    # Initialize an empty string to accumulate the response content
    answer = """"""
    for chunk in completion:
        # Append each chunk of content to the answer string
        answer += chunk.choices[0].delta.content or ""
        
    label_dict = extract_answer(answer)

    # Return the dictionary containing the generated questions
    return answer, label_dict


if __name__ == "__main__":
    df = pd.read_csv("../data/new_data/AI_feedback.csv", encoding='utf-8')
    comments = list(df['Comments'])
    random_number = random.randrange(0,165)
    answer, gen_label_dict = generate_labels(comments[random_number], GENERATE_LABELS_PROMPT, client)
    print("-"* 80)
    try:
        print(f"FEEDBACK: {gen_label_dict['feedback']}")
    except KeyError:
        print(f"FEEDBACK: {comments[random_number]}")
    print("-"* 80)
    try:
        print(f"AI Generated label: {gen_label_dict['label']}")
    except KeyError:
        print("Error in generation, rerunning process now. [process will rerun]")