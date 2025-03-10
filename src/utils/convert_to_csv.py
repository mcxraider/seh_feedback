import pandas as pd

def combine_xls(region):
    columns_to_read = ["Feedback id", "Feedback 1", "Feedback 2", "URL"]

    # Read in the two files. (Ensure your files are in a CSV format;
    # if they are truly Excel files, consider using pd.read_excel instead.)
    article = pd.read_excel(f"../data/official_data/feedback_{region}_article.xls", usecols=columns_to_read)
    course = pd.read_excel(f"../data/official_data/feedback_{region}_course.xls", usecols=columns_to_read)

    # Row bind the dataframes (i.e. concatenate them vertically)
    combined_df = pd.concat([article, course], ignore_index=True)
    combined_df.to_csv(f"../data/combined_data/feedback_{region}.csv", index=False)

def combine_xlsx(region):
    columns_to_read = ["Feedback id", "Feedback 1", "Feedback 2", "URL"]

    # Read in the two files. (Ensure your files are in a CSV format;
    # if they are truly Excel files, consider using pd.read_excel instead.)
    article = pd.read_excel(f"../data/official_data/feedback_{region}_article.xlsx", usecols=columns_to_read)
    course = pd.read_excel(f"../data/official_data/feedback_{region}_course.xlsx", usecols=columns_to_read)

    # Row bind the dataframes (i.e. concatenate them vertically)
    combined_df = pd.concat([article, course], ignore_index=True)
    combined_df.to_csv(f"../data/combined_data/feedback_{region}.csv", index=False)

def combine_region(region):
    xls_regions = ["VN", "SG"]

    if region in xls_regions:
        combine_xls(region)
    else:
        combine_xlsx(region)
