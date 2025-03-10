import pandas as pd
from groq import Groq
import json
import duckdb
import os
import time
import re
import sys
from tqdm import trange
from langchain_core.prompts import PromptTemplate
import logging
from math import ceil
from typing import List, Dict, Tuple
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

## load env variables
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
CHAT_MODEL   = os.environ["CHAT_MODEL"]
client       = Groq()
# get the date and time of the generation
now = datetime.today()
today_date = now.strftime('%Y-%m-%d_') + now.strftime('%I%M%p').lstrip("0")


GENERATE_EN_LABELS_PROMPT = '''
You are a linguistics professor with extensive experience in text analysis and classification. 
Your task is to categorise seller feedback for an article webpage on an e-commerce education platform.

Follow these steps carefully:
1. **Understand the Task**: Each feedback item must be assigned one or more labels from the following list:
   - 'Negative Complaint'
   - 'Constructive Criticism'
   - 'Design Feedback'
   - 'Positive Comment'
   - 'Genuine Question'
   - 'Unknown'

2. **Interpretation Guidelines**:
    - Negative Complaint - Expresses dissatisfaction. (E.g., "The UI is terrible and frustrating to use.")
    - Constructive Criticism – Offers feedback on what could be improved/ Positive feedback/ negative feedback that that I could use to improve the article.
    - Design Feedback – Mentions aspects related to visual design, user experience, or layout. (E.g., "The font is too small and hard to read.")
    - Positive Comment – Expresses satisfaction or praise. (E.g., "Great platform! I love using it.")
    - Genuine Question - The seller is asking a genuine question and is unsure about something. 
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

def load_region_data(region: str) -> pd.DataFrame:
    # Define the file path based on the region
    region_path = f"../data/combined_data/feedback_{region}.csv"

    # Load the data into a DataFrame
    try:
        df = pd.read_csv(region_path)
        
    except FileNotFoundError:
        print("\n\nERROR: Please ensure that you have followed the steps correctly and that the regions combined feedback is in the right folder and exists there.\n\n")
        sys.exit()
        
    # Filter out rows with missing or invalid data
    df_filtered = df[
        (df['Feedback 1'].notna()) &
        (df['Feedback 2'].notna()) &
        (df['Feedback 2'] != '{"description":""}')
    ].copy()  # Ensure df_filtered is a separate copy

    # Extract the 'description' field from JSON in 'Feedback 2'
    df_filtered.loc[:, 'Feedback 2'] = df_filtered['Feedback 2'].apply(
        lambda x: json.loads(x)['description'] if isinstance(x, str) else None
    )

    # Convert 'Feedback id' to numeric and drop rows with invalid IDs
    df_filtered.loc[:, 'Feedback id'] = pd.to_numeric(df_filtered['Feedback id'], errors='coerce')
    df_filtered = df_filtered.dropna(subset=['Feedback id'])

    return df_filtered


def format_llm_input(df: pd.DataFrame) -> Tuple[List[Dict[str, str]], Dict[int, str]]:
    # Extract feedback IDs and feedback text
    feedback_ids = list(df['Feedback id'])
    feedback_texts = list(df['Feedback 2'])
    feedback_urls = list(df['URL'])

    # Create a dictionary mapping feedback IDs to feedback text
    id_feedback = {int(feedback_id): [feedback,
                                      feedback_url] 
                   for feedback_id, feedback, feedback_url in zip(feedback_ids, feedback_texts, feedback_urls)}

    # Prepare the LLM input as a list of dictionaries
    llm_input = [{'id': feedback_id,
                  'feedback': feedback_ls[0]} for feedback_id, feedback_ls in id_feedback.items()]

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


def generate_labels(prompt, llm_input, num_per_batch):
    if not isinstance(llm_input, list):
        raise TypeError("llm_input must be a list")

    num_batches = ceil(len(llm_input) / num_per_batch)
    start_index = 0
    just_in_case_stop_index = 0
    total_tokens = 0
    labelled_data = []
    
    try:
        for i in trange(num_batches):
            end_index = min(start_index + num_per_batch, len(llm_input))
            batch_pairs = llm_input[start_index:end_index]
            
            try:
                batch_labels, tokens_used = generate_batch_labels(batch_pairs, prompt, client)
                total_tokens += tokens_used
            except ValueError:
                intermediate_end = min(start_index + 5, len(llm_input))
                batch_pairs = llm_input[start_index:intermediate_end]
                
                batch_labels, tokens_used = generate_batch_labels(batch_pairs, prompt, client)
                total_tokens += tokens_used

                intermediate_start = intermediate_end
                if intermediate_end < end_index:
                    batch_pairs = llm_input[intermediate_start:end_index]
                    batch_labels, tokens_used = generate_batch_labels(batch_pairs, prompt, client)
                    total_tokens += tokens_used


                if (i + 1) % 5 == 0:
                    print(f"Completed {i + 1} iterations. To prevent rate limits, sleeping for 60 seconds...")
                    time.sleep(60)

                start_index = intermediate_end  
                just_in_case_stop_index = intermediate_end
                continue  # Skip the rest of the loop
            
            labelled_data.extend(batch_labels)
            start_index = end_index

            if (i + 1) % 5 == 0:
                print(f"Completed {i + 1} iterations. To prevent rate limits, sleeping for 60 seconds...")
                time.sleep(60)

            just_in_case_stop_index = end_index
            time.sleep(2)

    except Exception as e:
        print(f"An error occurred while processing: {e}")
        print(f"Stopped at batch {just_in_case_stop_index}\n")
        sys.exit()
        
    return total_tokens, labelled_data
        
        
def pair_id_feedback(id_feedback, feedback_labels: list):
    for i in range(len(feedback_labels)):
        # get the feedback id
        id = feedback_labels[i]['feedback_id']
        feedback_labels[i]['Comment'] = id_feedback[id][0]
        feedback_labels[i]['URL'] = id_feedback[id][1]
        
    return feedback_labels


def process_output(combined, org_df, pattern=r"/([^/]+)/(\d+)"):
    # Convert to a DataFrame and rename columns
    combined_df = pd.DataFrame(combined)
    combined_df.rename(columns={
        'feedback_id': 'Feedback id',
        'Comment': 'Comments',
        'label': 'Label(s)',
        'URL': 'Link to Article'
    }, inplace=True)

    # Clean up the Comments column
    combined_df['Comments'] = (
        combined_df['Comments']
        .str.replace('""', '"', regex=False)
        .str.strip('"')
        .str.replace('\n', '')
    )

    # Function to extract text type and article number dynamically from URL
    def extract_text_and_number(url):
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)
        return "NIL", "NIL"

    # Apply the extraction to the "Link to Article" column
    combined_df[['Type', 'Article ID']] = combined_df['Link to Article'].apply(
        lambda url: pd.Series(extract_text_and_number(url))
    )

    # Reorder the columns for combined_df
    desired_order = ['Feedback id', 'Article ID', 'Comments', 'Label(s)', 'Link to Article', 'Type']
    combined_df = combined_df[desired_order]

    # Inner join with org_df on "Feedback id" using only the "Feedback 1" column
    combined_df = pd.merge(combined_df, org_df[['Feedback id', 'Feedback 1']],
                             on='Feedback id', how='inner')

    # Reorder to append "Feedback 1" to the front of the desired order
    desired_order_extended = ['Article ID', 'Feedback 1', 'Comments', 'Label(s)', 'Link to Article', 'Type']
    combined_df = combined_df[desired_order_extended]
    
    # Convert any list entries in 'Label(s)' to a string for sorting purposes.
    combined_df['Label(s)'] = combined_df['Label(s)'].apply(
        lambda x: ', '.join(x) if isinstance(x, list) else x
    )
    
    # Optionally convert 'Article ID' to numeric for proper numeric sorting.
    combined_df['Article ID'] = pd.to_numeric(combined_df['Article ID'], errors='coerce')
    
    # Sort by 'Article ID' and then by 'Label(s)'
    combined_df.sort_values(by=['Article ID', 'Label(s)'], inplace=True)

    return combined_df

def export_to_csv(final_df, path):    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    query = '''
    WITH cte AS (
        SELECT "Article ID", COUNT(*) AS feedbacks_per_article
        FROM df
        GROUP BY "Article ID"
        ORDER BY "Article ID"
    )
    SELECT *
    FROM df
    INNER JOIN cte ON cte."Article ID" = df."Article ID"
    ORDER BY feedbacks_per_article DESC;
    '''
    
    # Execute the SQL query with DuckDB, passing the dataframe as a parameter.
    result = duckdb.query(query, {'df': final_df}).final_df()
    
    # Drop the extra column if it exists.
    if "Article ID_1" in result.columns:
        result.drop("Article ID_1", axis=1, inplace=True)
    
    # Overwrite or create the file
    final_df.to_csv(path, index=False, mode='w', encoding='utf-8')


def en_pipeline(region):
    
    CSV_OUTPUT_LOCATION = f'../data/labelled_feedback/{today_date}_{region}_labelled_feedback_data.csv'
    
    df = load_region_data(region)
    print("df loaded")
    sys.exit()
    llm_input, id_feedback = format_llm_input(df)
    
    # Plan on what to do with this token consumed.
    total_tokens_consumed, feedback_labels = generate_labels(GENERATE_EN_LABELS_PROMPT, llm_input, num_per_batch=10)
    combined = pair_id_feedback(id_feedback, feedback_labels)
    final_df = process_output(combined, df)
    export_to_csv(final_df, CSV_OUTPUT_LOCATION)
    print(f"\n\nThis operation run required {total_tokens_consumed} tokens\n\n")
    return total_tokens_consumed

