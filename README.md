# OpenPDF-GPT
<div align="center">


![Demo](https://github.com/Joseph-M-Cook/OpenPDF-GPT/blob/48ab065ed82ee40115edc79ba08e5c6abdd52009/OpenPDF-GPT.png)
  </div>
  
# Description
OpenPDF-GPT is an advanced tool for PDF data extraction and knowledge comprehension. It leverages state-of-the-art technologies in Computer Vision (CV), Large Language Models (LLMs), and vector embeddings to convert your PDFs into structured, query-able data. It goes beyond conventional OCR methods by not only extracting textual content but also tabular data and information hidden in complex layouts. Moreover, it also enables advanced analysis and insights by supporting tabular data extraction.

# Features
- **Textual Data Extraction**: Extracts plain text from PDF files leveraging PyPDF2.
- **Tabular Data Extraction**: Extracts tables from PDF files using Camelot.
- **Data Comprehension and Summarization**: Uses OpenAI's GPT-4 for understanding and summarizing the extracted data.
- **Data Indexing**: Indexes the summarized data into Pinecone for semantic search capabilities.
- **Query Resolution**: Uses a question-answering chain for responding to queries about the extracted data.

# References
- [pdfGPT](https://github.com/bhaskatripathi/pdfGPT) by [bhaskatripathi](https://github.com/bhaskatripathi)
- [GPT-4: How to Chat With Multiple PDF Files](https://www.youtube.com/watch?v=Ix9WIZpArm0&t=12s) by [chatwithdata] on YouTube
