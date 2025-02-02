# MedLLM: LLM Chatbot for Medical Reports Analysis

## By Dun Li Chan 

MedLLM is a chatbot built using Streamlit and JamAI that allows users to interact with uploaded medical PDFs, querying and analyzing their medical report.

## Features
- Upload PDF medical reports to create a Knowledge Base.
- Ask questions and receive answers based on your uploaded report.
- Customize model parameters like temperature and top_p.

## Requirements
- Python 3.7+
- Streamlit
- JamAI API Key and Project ID

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/medllm.git
   cd medllm

2. Create a virtual environment (optional):
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
3. Install dependencies:
    ```bash
    pip install -r requirements.txt