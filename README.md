AI Text Checker 

This project provides a graphical user interface (GUI) application designed to detect AI-generated text. The app leverages various text analysis techniques to assess whether a given text is likely created by an AI model.

Features
Text Analysis: The app evaluates text based on perplexity, entropy, n-gram frequencies, and TF-IDF cosine similarity.
User Interface: A Tkinter-based GUI allows you to select files (TXT, PDF, DOCX), choose the text language, and set the detection threshold.
Preferences: User preferences for language and threshold are saved and can be loaded for future use.
Getting Started
To run this application on a fresh Python installation, follow these instructions:

Prerequisites
Python: Make sure Python 3.7 or newer is installed. You can download it from python.org.

Install Required Packages: Install the necessary Python libraries using pip. Open your terminal or command prompt and execute:

pip install nltk torch transformers scikit-learn pdfplumber python-docx
Download NLTK Resources: The script uses NLTK for text processing. Download the required resources by running these Python commands:

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
Running the Script

Clone the Repository:

git clone https://github.com/YunoStoleMySyntax/AI-textcheck.git
cd AI-textcheck
Save and Run the Script:
Save the provided script as ai_text_detector.py in the cloned repository folder. Then, run it using:

python Is-it-AI-text.py

Using the Application:

Browse for File: Click the "Browse" button to select a text file (.txt, .pdf, or .docx).
Select Language: Choose the language of the text from the dropdown menu.
Set Threshold: Enter a numeric threshold value for AI detection.
Analyze: Click "Analyze" to process the text and view the results.
License
This project is licensed under the MIT License - see the LICENSE file for details.

Contact
For any issues or questions, please open an issue on the GitHub repository.

Feel free to customize any section as needed!
