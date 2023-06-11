# OpenPDF-GPT
<div align="center">


![Demo](https://github.com/Joseph-M-Cook/OpenPDF-GPT/blob/48ab065ed82ee40115edc79ba08e5c6abdd52009/OpenPDF-GPT.png)
  </div>
  
# Description
OpenPDF-GPT is an advanced data extraction and comprehension tool for PDF documents. This project leverages Computer Vision, Large Language Models (LLMs), and vector embeddings to revolutionize data extraction from PDF files. Not only does it extract and understand the content of PDFs, but it also supports tabular data, allowing for advanced analysis and unlocking of new insights. The main highlight of this project is its ability to integrate the understanding capabilities of OpenAI's GPT-4 language model into the data extraction process, providing more contextually accurate results.

# Features
- **Textual Data Extraction**: Extracts plain text from PDF files leveraging PyPDF2.
- **Tabular Data Extraction**: Extracts tables from PDF files using Camelot.
- **Data Comprehension and Summarization**: Uses OpenAI's GPT-4 for understanding and summarizing the extracted data.
- **Data Indexing**: Indexes the summarized data into Pinecone for semantic search capabilities.
- **Query Resolution**: Uses a question-answering chain for responding to queries about the extracted data.
