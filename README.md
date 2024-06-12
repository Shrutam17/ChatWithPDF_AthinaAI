# ChatwithPDF
ChatwithPDF-Your document assistant.
Demo Link :- [Link] (https://drive.google.com/file/d/1ix6_7Yj9M33Frh6eaWGhh9ValcIGpJyl/view?usp=drive_link)

# Introduction 
Here I have designed a RAG (Retrieval-Augmented Generation) powered
chatbot using the PDF document as the knowledge source. The chatbot answers
the questions related to the contents of PDF. The PDF here is Churchill Motor
Insurance Policy Booklet. In this project, I utilized HuggingFace's instructor-xl
model for embeddings and Google's flan-t5-large model for conversational tasks,
ensuring robust and accurate responses

# Tech Stack 
* Python: Core programming language for development.
* Streamlit: UI development tool for creating interactive web applications.
* PyPDF2, python-docx, pptx: Libraries for parsing and extracting text from document formats.
* PyTesseract: Optical character recognition (OCR) tool for extracting text from images.
* Transformers (Hugging Face): Library for natural language processing (NLP) tasks, including question-answering and image captioning.
* PyTorch: Deep learning framework for tasks such as image captioning using pre-trained models.
* NLTK: Toolkit for natural language processing tasks like tokenization and part-of-speech tagging.
* Hugging Face's google-flan-t5-large: Language model for generating responses to user queries and conducting conversations.
* Hugging Face's: instructor-xl model for embeddings
* Langchain: Library for advanced text processing tasks such as text splitting, embeddings, and conversational chains.


# External Dependencies 
Ensure the following dependencies are installed externally:
* Tesseract OCR: Install Tesseract OCR for PyTesseract to work properly. (Install via: sudo apt install tesseract-ocr)

# Installation Instructions
1. Create a virtual environment with Python 3.9:
  ** conda create -p venv python==3.9
2. Activate the virtual environment:
  ** conda activate venv
3. Install the required Python packages using pip:
  ** pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub InstructorEmbedding sentence_transformers==2.2.2 tiktoken aspose.slides spire.pdf python-pptx python-docx
4. Install external dependencies:
  ** sudo apt install tesseract-ocr ``

# Usage
1. Run the Streamlit app:
 ** streamlit run app.py
2. Upload document in PDF format.
3. Click on the "Process" button to analyze the uploaded document.
4. Ask questions related to the content of the document in the text input box.
5. The chatbot will analyze the documents and provide answers based on the content.
6. Download the question-answer history in the form of a .txt file by clicking the download button.

# Note 
* Ensure that the uploaded documents contain document with readable text. The effectiveness of the chatbot depends on the quality and relevance of the content provided.
* Prompts can be in natural language but must be proper and chatbot is capable of answering indirect queries as well.
* The project depends on Hugging Face's google flan and instructor-xl for conversational chain and embeddings respectively.
* .env file needs to be created with  API keys or virtual environment needs to  be created.

# Results
* Out of 32 pairs of human response and chatbot response against corresponding query, 25 of them have BERT similarity score over 3.0 on a scale of 1 to 5 and hence approximate accuracy of 80%.


# Conclusion 
This chatwithPDF provides convenience to the user to summarize the entire document and ask questions regarding the document rather than reading entire which is very useful. 
