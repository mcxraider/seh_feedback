import pandas as pd
from groq import Groq
import json
import os
import time
import re
from tqdm import trange
from langchain_core.prompts import PromptTemplate
import logging
from math import ceil
from typing import List, Dict, Tuple, Any

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

## load evv variables
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHAT_MODEL   = os.environ["CHAT_MODEL"]
client       = Groq()

GENERATE_EN_LABELS_PROMPT = '''
You are an linguistics professor tasked with classifying seller feedback for an e-commerce platform. 
Each feedback item should be categorised into one or more appropriate labels from the following list:
['Negative Complaint','Constructive Criticism','Design Feedback','Positive Comment','Neutral']
You are not to write any code, but just use your knowledge to classify the feedback.
Your output should be the feedback IDs and their corresponding label.

Now classify the following feedback:
Feedbacks: {pairs}

Example Output format:
[{{"feedback_id": 123456, "label": ["Negative Complaint"]}}, {{"feedback_id": 423456, "label": ["Constructive Criticism","Design Feedback"]}}, {{"feedback_id": 654321, "label": ["Negative Complaint"]}}]

Double check and ensure that your format output matches the example output format provided.
''' 

def load_region_data(region: str) -> pd.DataFrame:
    # Define the file path based on the region
    region_path = f"../data/official_data/feedback_{region}.xls"

    # Specify columns to read
    columns_to_read = ["Feedback id", "Feedback 1", "Feedback 2"]

    # Load the data into a DataFrame
    df = pd.read_excel(region_path, usecols=columns_to_read)

    # Filter out rows with missing or invalid data
    df_filtered = df[
        (df['Feedback 1'].notna()) &
        (df['Feedback 2'].notna()) &
        (df['Feedback 2'] != '{"description":""}')
    ]

    # Extract the 'description' field from JSON in 'Feedback 2'
    df_filtered['Feedback 2'] = df_filtered['Feedback 2'].apply(
        lambda x: json.loads(x)['description'] if isinstance(x, str) else None
    )

    # Convert 'Feedback id' to numeric and drop rows with invalid IDs
    df_filtered['Feedback id'] = pd.to_numeric(df_filtered['Feedback id'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Feedback id'])

    return df_filtered


def format_llm_input(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], Dict[int, str]]:
    # Extract feedback IDs and feedback text
    feedback_ids = list(df['Feedback id'])
    feedback_texts = list(df['Feedback 2'])

    # Create a dictionary mapping feedback IDs to feedback text
    id_feedback = {int(feedback_id): feedback for feedback_id, feedback in zip(feedback_ids, feedback_texts)}

    # Prepare the LLM input as a list of dictionaries
    llm_input = [{'id': feedback_id, 'feedback': feedback} for feedback_id, feedback in id_feedback.items()]

    return llm_input, id_feedback


def get_id_labels(llm_response: str, pattern: str = r'\[\s*\{(?:.|\n)*\}\s*\]') -> List[Dict[str, str]]:
    if not isinstance(llm_response, str):
        raise TypeError("The LLM response must be a string.")

    try:
        # Find the match
        match = re.search(pattern, llm_response, re.DOTALL)
        if not match:
            print(f"THIS RESPONSE WAS PRODUCED AND WAS UNABLE TO BE PICKED UP:\n{llm_response}")
            raise ValueError("No valid JSON list found in the response.")

        json_string = match.group(0)
        result = json.loads(json_string)

        # Validate the structure of the result
        if not isinstance(result, list) or not all(isinstance(item, dict) for item in result):
            raise ValueError("Extracted JSON is not a list of dictionaries.")
        
        return result

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to decode JSON: {e}")
    except Exception as e:
        raise RuntimeError(f"An unexpected error occurred: {e}")


def generate_batch_labels(id_feedback_pairs, label_prompt: str, client):
    prompt = PromptTemplate(
        template=label_prompt,
        input_variables=["pairs"],
    )

    final_prompt = prompt.format(pairs=id_feedback_pairs)

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
        top_p=1  # Nucleus sampling parameter (1 means only the most likely tokens are considered)
    )

    # Initialize an empty string to accumulate the response content
    response = completion.choices[0].message.content

    tokens_used = completion.usage.total_tokens
    pairings = get_id_labels(response)
    
    return pairings, tokens_used


def generate_labels(llm_input, num_per_batch, output_file=f'../data/llm_responses/llm_responses.json'):
    # Determine the number of batches
    num_batches = ceil(len(llm_input) / num_per_batch)

    # Initialise indices for batch processing
    start_index = 0

    just_in_case_stop_index = 0
    
    total_tokens = 0
    
    try:
        for i in trange(num_batches):
            # Calculate the batch indices
            end_index = start_index + num_per_batch
            batch_pairs = llm_input[start_index:end_index]
            try:
                
                # Call the function to generate labels for the current batch
                batch_labels, tokens_used = generate_batch_labels(batch_pairs, GENERATE_EN_LABELS_PROMPT, client)
                total_tokens += tokens_used
            except ValueError:
                    intermediate_end = start_index+5
                    batch_pairs = llm_input[start_index:intermediate_end]
                    batch_labels, tokens_used = generate_batch_labels(batch_pairs, GENERATE_EN_LABELS_PROMPT, client)   
                    total_tokens += tokens_used

                    # Write the current batch to the JSON file
                    with open(output_file, 'a') as json_file:
                        # Convert the batch to a JSON string and write it
                        for label in batch_labels:
                            json_file.write(json.dumps(label) + '\n')
                    
                    intermediate_start = intermediate_end
                    batch_pairs = llm_input[intermediate_start:end_index]
                    batch_labels, tokens_used = generate_batch_labels(batch_pairs, GENERATE_EN_LABELS_PROMPT, client)   
                    total_tokens += tokens_used

                    # Write the current batch to the JSON file
                    with open(output_file, 'a') as json_file:
                        # Convert the batch to a JSON string and write it
                        for label in batch_labels:
                            json_file.write(json.dumps(label) + '\n')
                    
                    # Sleep for 60 seconds every 10 iterations
                    if (i + 1) % 5 == 0:
                        print(f"Completed {i + 1} iterations. To prevent rate limits, sleeping for 60 seconds...")
                        time.sleep(60)
                        
                    continue
                
            # Update the start index for the next batch
            start_index = end_index

            # Write the current batch to the JSON file
            with open(output_file, 'a') as json_file:
                # Convert the batch to a JSON string and write it
                for label in batch_labels:
                    json_file.write(json.dumps(label) + '\n')

            # Sleep for 60 seconds every 10 iterations
            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1} iterations. To prevent rate limits, sleeping for 60 seconds...")
                time.sleep(60)
                
            just_in_case_stop_index = end_index
            # Include  extra rest (not sure why but just in case lol)
            time.sleep(2)

        print(f"All batches written to {output_file}")
        return total_tokens

    except Exception as e:
        
        print(f"An error occurred while processing: {e}")
        print(f"Stopped at batch {just_in_case_stop_index}\n")


def read_json_file(file_path: str):
    try:
        data = []
        with open(file_path, 'r') as json_file:
            for line in json_file:
                data.append(json.loads(line.strip()))
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        
        
def pair_id_feedback(id_feedback: dict, feedback_labels: list):
    for i in range(len(feedback_labels)):
        id = feedback_labels[i]['feedback_id']
        comment = id_feedback[id]
        feedback_labels[i]['Comment'] = comment
        
    return feedback_labels


def write_to_csv(region: str, combined):
    # Convert to a DataFrame
    combined_df = pd.DataFrame(combined)

    # Rename columns to match the required format
    combined_df.rename(columns={'feedback_id': 'Feedback id',
                                'label': 'Label',
                                'Comment': 'Comment'},
                        inplace=True)
    combined_df['Comment'] = combined_df['Comment'].str.replace('""', '"', regex=False).str.strip('"')

    # Save to CSV
    csv_filename = f'../data/labelled_feedback/{region}_labelled_feedback_data.csv'
    combined_df.to_csv(csv_filename, index=False)
    print(f"{region} Labels are now store in the data folder under labelled_feedback.")


def main():
    region = "SG"
    df = load_region_data(region)
    llm_input, id_feedback = format_llm_input(df)
    # Plan on what to do with this token consumed.
    total_tokens_consumed = generate_labels(llm_input, num_per_batch=10)
    feedback_labels = read_json_file(file_path=f'../data/llm_responses/{region}_llm_responses.json')
    combined = pair_id_feedback(id_feedback, feedback_labels)
    write_to_csv(region, combined)
    print(f"This operation run has used up {total_tokens_consumed} tokens")
    return total_tokens_consumed



if __name__ == "__main__":
    total_tokens_used = main()
