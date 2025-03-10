# import the main functions from the other scripts
from utils.convert_to_csv import combine_region
from utils.english_pipeline import en_pipeline
import os

# define region u want to gather analysis on.
REGION = "SG"

def main(region):
    
    if f"feedback_{region}.csv" not in os.listdir("../data/combined_data"):
        combine_region(region)
    
    if region in ["SG"]:
        tokens_used = en_pipeline(region)
    else :
        print("pipeline for this region has not yet been built")
        
    return tokens_used
    
    
if __name__ == '__main__':
    total_tokens = main(REGION)




