# Chat-with-PDF
Chat with PDF is a Streamlit application that allows users to upload PDF and DOCX files and interact with their content through an AI-powered chatbot. Leveraging OpenAI's GPT-3.5-turbo and LangChain, this app provides a seamless way to query and retrieve information from uploaded documents.

# Features
Upload Files: Supports both PDF and DOCX file formats.
Text Extraction: Extracts text from the uploaded files using PyPDF2 for PDFs and python-docx for DOCX files.
Text Chunking: Splits extracted text into manageable chunks for better processing.
Vector Store Creation: Utilizes HuggingFaceEmbeddings and FAISS for vector storage and retrieval.
Conversational Interface: Allows users to ask questions about the content of their files and receive responses in real-time.
Session Memory: Maintains conversation history to provide contextually relevant responses.
Cost Tracking: Displays token usage and cost for interactions using the OpenAI API.

# Installation
https://github.com/attaelahi/Chat-with-PDF.git

cd chat-with-pdf

pip install -r requirements.txt

streamlit run app.py

# Usage
Open the application in your web browser.
Upload your PDF or DOCX files using the sidebar.
Enter your OpenAI API key.
Click "Process" to analyze the documents.
Ask questions about the content of your files and get instant responses.

# Screenshots
<img width="960" alt="image 0" src="https://github.com/attaelahi/Chat-with-PDF/assets/72361631/e5c3e61a-b188-4e87-92dd-0b5f6adfc07e">
<img width="960" alt="image 1" src="https://github.com/attaelahi/Chat-with-PDF/assets/72361631/48a38bc3-1c6c-4123-966a-b5290c90aa47">

# Contributing
Contributions are welcome! Please open an issue or submit a pull request for any changes or additions.

