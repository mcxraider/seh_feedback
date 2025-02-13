import os
import pandas as pd
import win32com.client


def convert_xls(directory: str):
    """
    Converts all .xls files in the specified directory to .xlsx format and deletes the .xls files.
    
    :param directory: The directory containing .xls files.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.xls')]
    
    excel = win32com.client.Dispatch("Excel.Application")
    excel.Visible = False  # Run in the background
    
    for fname in files:
        xls_file_path = os.path.abspath(os.path.join(directory, fname))
        xlsx_file_path = os.path.abspath(os.path.join(directory, fname + "x"))  # Append 'x' to make .xlsx
        
        wb = excel.Workbooks.Open(xls_file_path)
        wb.SaveAs(xlsx_file_path, FileFormat=51)  # 51 = xlsx format
        wb.Close()
        
        os.remove(xls_file_path)
        print(f"Converted and removed: {fname}")

    excel.Quit()


def combine_feedback_sheets(directory: str):
    """
    Combines feedback files for each region (VN, PH, ID, MY, SG) 
    into a single consolidated file and deletes the original individual files.
    
    :param directory: The directory containing the feedback files.
    """
    files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

    # Group files by region (VN, PH, ID, MY, SG)
    region_files = {}
    for file in files:
        parts = file.replace('.xlsx', '').split('_')
        if len(parts) > 2:  # Ensures files have 'Article' or 'Course'
            region = parts[1]
            if region not in region_files:
                region_files[region] = []
            region_files[region].append(file)

    # Process each region
    for region, file_list in region_files.items():
        combined_df = pd.DataFrame()

        for file in file_list:
            file_path = os.path.join(directory, file)
            df = pd.read_excel(file_path)
            combined_df = pd.concat([combined_df, df], ignore_index=True)

        # Save merged file
        output_filename = f"feedback_{region}.xlsx"
        output_path = os.path.join(directory, output_filename)
        combined_df.to_excel(output_path, index=False)
        print(f"Saved: {output_filename}")

        # Delete original files
        for file in file_list:
            os.remove(os.path.join(directory, file))
            print(f"Deleted: {file}")

if __name__ == "__main__":
    official_data_dir = "../data/official_data"

    # Convert .xls files to .xlsx first
    convert_xls(official_data_dir)

    # Then merge feedback files
    combine_feedback_sheets(official_data_dir)
