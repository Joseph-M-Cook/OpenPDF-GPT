# OpenPDF-GPT
<div align="center">


![Demo](https://github.com/Joseph-M-Cook/OpenPDF-GPT/blob/facc24828f754cbc0e7a3e4731fdd3a83f859572/OpenPDF-GPT-README.png)
  </div>
  
## Description
OpenPDF-GPT is an advanced tool for PDF data extraction and knowledge comprehension. It leverages state-of-the-art technologies in Computer Vision (CV), Large Language Models (LLMs), and vector embeddings to convert your PDFs into structured, natural language query-able data. It goes beyond conventional OCR methods by not only extracting textual content but also tabular data and information hidden in complex layouts, enabling advanced analysis and insights unlike any other system, solution, software, or framework.

## Features
- **Textual Data Extraction**: Extracts plain text from PDF files leveraging [PyPDF2](https://pypi.org/project/PyPDF2/).
- **Tabular Data Extraction**: Extracts tables from PDF files using [Camelot](https://pypi.org/project/camelot-py/).
- **Data Comprehension and Summarization**: Uses OpenAI's [GPT-4](https://openai.com/research/gpt-4) for understanding and summarizing the extracted data.
- **Embeddings**: Uses OpenAI's [text-embedding-ada-002](https://platform.openai.com/docs/guides/embeddings) to embed chunks of extracted data.
- **Data Indexing**: Indexes the summarized data into [Pinecone](https://www.pinecone.io/) for semantic search capabilities.
- **Query Resolution**: Uses a [Langchain](https://python.langchain.com/en/latest/index.html) question-answering chain for responding to queries about the extracted data.

## References
- [pdfGPT](https://github.com/bhaskatripathi/pdfGPT)
- [GPT-4: How to Chat With Multiple PDF Files](https://www.youtube.com/watch?v=Ix9WIZpArm0&t=12s)
- [OpenAI Embeddings API - Searching Financial Documents](https://www.youtube.com/watch?v=xzHhZh7F25I)
- [Table Question Answering](https://docs.pinecone.io/docs/table-qa)
- [PDF-chat-app](https://github.com/sukumar18/PDF-chat-app/blob/main/pdfapp.py)
- [LangChain101: Question A 300 Page Book (w/ OpenAI + Pinecone)](https://www.youtube.com/watch?v=h0DHDp1FbmQ&t=482s)

 ## License
OpenPDF-GPT is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.
