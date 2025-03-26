# SEH Feedback analysis for SG Region

## 📝 Step-by-Step: How to Prepare Files and Run the Project

### 📥 Part 1: Prepare the Feedback Files

1. **Download SG Feedback Files**
   - Go to the **CMS Feedback Centre**.
   - Download feedback files, but **filter for only**:
     - **Type: Article**
     - **Type: Course**

2. **You should now have 2 separate files:**
   - One file with feedback for **articles**
   - One file with feedback for **courses**

3. **Rename the files:**
   - Rename the article feedback file to: `feedback_SG_article`
   - Rename the course feedback file to: `feedback_SG_course`

4. **Move both files into the project folder:**
   - Place them inside this folder path (relative to the project root):
     ```
     data/official_data/
     ```
     -> Put them in the official_data folder that's inside data.
     -> If there are existing files inside, make sure to delete them!


## 🚀 Part 2: Run the Project in Terminal

Follow these steps to run the project on your Windows machine:

### 1. 📂 Open your Terminal
You can use **Command Prompt**, **PowerShell**, or **Windows Terminal**.

### 2. 📁 Navigate to the Project Folder
In your terminal, run the following command:
```bash
cd Desktop/seh/
```
✅ This command changes your current directory to where your project is located. Adjust the path if your project is stored elsewhere.


### 3. 📂 Navigate to the `src` Folder
```bash
cd src
```
✅ This moves you into the folder where your main script is located.

### 4. ▶️ Run the Project
Now start the app by running:
```bash
python main.py
```


### ❗️Tips
- If `python main.py` doesn’t work, try:
  ```
  python3 main.py
  ```
- If you see an error saying Python is not found, you might need to install it. Let me know if you need help with that!

