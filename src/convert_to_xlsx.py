import win32com.client
import os

def convert_xls(files):
    
    for fname in files:
        # Convert relative path to absolute path
        xls_file_path = os.path.abspath(f"../data/official_data/{fname}")
        xlsx_file_path = os.path.abspath(f"../data/official_data/{fname}x")

        excel = win32com.client.Dispatch("Excel.Application")
        wb = excel.Workbooks.Open(xls_file_path)
        wb.SaveAs(xlsx_file_path, FileFormat=51)  # 51 = xlsx format
        wb.Close()
        excel.Quit()
        
        os.remove(xls_file_path)
        


if __name__ == "__main__":
    filenames = os.listdir("../data/official_data")
    convert_xls(filenames)
