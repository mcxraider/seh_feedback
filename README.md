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

### 📥 Part 2: Run the project in Terminal

1. **Open your Terminal**

2. **Navigate to the project folder**
   - In the terminal window, type this command:
     ```
     cd Desktop/seh/src
     ```
   - Then press `Enter`.

   ✅ This tells your computer to “Change Directory” to where your project is located.

3. **Run the project**
   - Type the following command and press `Enter`:
     ```
     python main.py
     ```

   ✅ This will start the project (make sure Python is installed!).
---

### ❗️Tips
- If `python main.py` doesn’t work, try:
  ```
  python3 main.py
  ```
- If you see an error saying Python is not found, you might need to install it. Let me know if you need help with that!


