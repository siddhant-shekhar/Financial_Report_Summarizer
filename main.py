# importing the api_key saved in api_key.py file
from dotenv import load_dotenv
import os
 
# streamlit for ui
import streamlit as st
 
# for data extraction from pdf
import pdfplumber #for text
import fitz  # PyMuPDF (for image/chart)
import base64 # for image/chart conversion
 
from docx import Document
from docx.shared import RGBColor
from docx.shared import Pt
from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
 
# for nlp
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
 
 
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
 
# for api calls and langchain implementation
from openai import OpenAI
import requests
import io
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
 
# Ensure NLTK downloads
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
 
#Extract the API KEY from the OS variable
load_dotenv()
api_key=os.getenv("OPENAI_API_KEY")
 
 
 
#Mention the Specific version of the OpenAI GPT 
MODEL="gpt-4o"
 
# Setting OpenAI API key as an environment variable for security
client = OpenAI(
   
    api_key=api_key
)
 
# Send image encoded in base64 to OpenAI GPT-4 model for text generation.
def send_to_gpt(base64_image, api_key):
 
    # specifying content type and authorization
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
        
    }
 
    # defining payload
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What’s in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    try:
        # storing and returning the response we got from the openai
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, verify=False)
        response.raise_for_status()  # Raise an error for non-2xx status codes
        return response.json()['choices'][0]['message']['content']
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error sending image to GPT-4 model: {e}")
        return "Error: Unable to generate text from image"
 
# Extract text and images from each page of a PDF document.
def extract_text_and_images(pdf_path):
    text_pages = []
    images = []
    seen_xrefs = set()
 
    try:
        # using pdfplumber to extract text data from the pdf file
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text_pages.append({
                    "page_num": page_num,
                    "content": page.extract_text()
                })
        
        # using fitz to extract images/charts from the pdf
        doc = fitz.open(pdf_path)
        for page_num in range(doc.page_count):
            page = doc[page_num]
            image_list = page.get_images(full=True)
 
            for img_index, img in enumerate(image_list):
                xref = img[0]
 
                if xref in seen_xrefs:
                    continue
 
                seen_xrefs.add(xref)
                image_data = doc.extract_image(xref)
                images.append({
                    "page_num": page_num,
                    "img_index": img_index,
                    "image_data": image_data["image"]
                })
    except Exception as e:
        st.error(f"Error extracting text and images from PDF: {e}")
 
    return text_pages, images
 
 
# Integrate extracted PDF text with GPT-4 generated responses for images.
def integrate_text_and_gpt_responses(text_pages, images, api_key, doc_num):
    integrated_text = []
 
    # Initialize a counter for the current image index
    img_index = 0
 
    # Iterate through each page's text and insert GPT responses where necessary
    for text_page in text_pages:
        page_num = text_page["page_num"]
        text_content = text_page["content"]
 
        # Add text content to integrated_text
        integrated_text.append(text_content)
 
        # Check if there are images for the current page
        while img_index < len(images) and images[img_index]["page_num"] == page_num:
            image_data = images[img_index]["image_data"]
            base64_image = base64.b64encode(image_data).decode('utf-8')
 
            # Send image to GPT-40 model to obtain generated text
            gpt_response = send_to_gpt(base64_image, api_key)
            # Insert GPT response at the appropriate position in integrated_text
            integrated_text.append(gpt_response)
 
            # Move to the next image
            img_index += 1
 
    return integrated_text
 
 
# Clean and preprocess text by removing special characters, tokenizing, and lemmatizing.
def preprocess_text(text):
    try:
        # Remove special characters except commas and periods
        text = re.sub(r'[^\w\s.,]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize the text into words
        words = word_tokenize(text)
        # Initialize WordNetLemmatizer
        lemmatizer = WordNetLemmatizer()
        # Lemmatize each word
        cleaned_words = [lemmatizer.lemmatize(word) for word in words]
        # Load stopwords
        stop_words = set(stopwords.words('english'))
        # Filter out stopwords
        cleaned_words = [word for word in cleaned_words if word.lower() not in stop_words]
        # Join cleaned words back into text
        cleaned_text = ' '.join(cleaned_words)
 
        return cleaned_text
    except Exception as e:
        st.error(f"Error preprocessing text: {e}")
        return "Error: Unable to preprocess text"
 
# Extract text from a PDF file, integrate with GPT-4 responses, and preprocess.
def extract_text_from_file(uploaded_file, api_key, idx):
    try:
        text_pages, images = extract_text_and_images(uploaded_file)
        integrated_text = integrate_text_and_gpt_responses(text_pages, images, api_key, idx + 1)
        cleaned_text = preprocess_text("\n\n".join(integrated_text))
        return cleaned_text
    except Exception as e:
        st.error(f"Error extracting text from file: {e}")
        return "Error: Unable to extract text from file"
 
 
# function to process the merged text of different files into the chunks and knowledge base
def process_text(text):
    # Initialize text splitter to split text into chunks
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size= 32000,
        chunk_overlap= 500,
        length_function= len
    )
    # Split the text into chunks
    chunks = text_splitter.split_text(text)

    # Load a model for generating embeddings from Huggingface
    embeddings = OpenAIEmbeddings()

    # Create a FAISS index from the text chunks using the embeddings
    knowledgeBase = FAISS.from_texts(chunks, embeddings)


    return knowledgeBase

################################################################################################
 
# prompt of 2-pager summary
prompt1 = (
    '''Generate a 2-pager summary (around 400-500 words) for the following 10 queries followed by the instructions to be followed while creating the summary.: 1. Business Overview  Include information such as the company's formation and incorporation date (only if available in the text), headquarters location, business description, employee count, latest revenues, stock exchange listing and market capitalization, number of offices and locations, and details on their clients/customers. 2. Business Segment Overview. i. Extract the revenue percentage of each component (verticals, products, segments, and sections) as a part of the total revenue. ii. Performance: Evaluate the performance of each component by comparing the current year's sales/revenue and market share with the previous year's numbers.  iii. Sales Increase/Decrease explanation: Explain the causes of the increase or decrease in the performance of each component.  4. Breakdown of sales and revenue by geography, specifying the percentage contribution of each region to the total sales. 5. Summarize geographical data, such as workforce, clients, and offices, and outline the company's regional plans for expansion or reduction. 6. Analyze and explain regional sales fluctuations, including a geographical sales breakdown to identify sales trends. 7. Year-over-year sales increase or decline and reasons for the change. 8. Summary of rationale & considerations (risks & mitigating factors). 9. SWOT Analysis. 10. Information about credit rating/credit rating change/change in the rating outlook. The generated text should have proper heading, subheading and content aligned. The heading should be 'Two Pager Summary for' followed by the company name.\nUse less bullet points, provide the summary in paragrapghs, use bullet points only when very much required. \nTry not to include any conclusion in the last and also do not include any information except my queries (no conclusion, no information, or instructions).\nTry to answer the queries from the text i have provided that is dont return any answer outside the context of the text provided.\n Give information only from the text provided. \n If possible, give summary in paragraphs, use bullet points only when needed. stick to the wrod limit of 400-500'''
)
 
 
# Function to generate a two-pager summary based on given texts
def two_pager_summary(texts):
 
    for idx, text in enumerate(texts):
        # Process and extract text to create a knowledge base
        knowledgeBase = process_text(text)

        # Perform similarity search in the knowledge base using prompt 1
        docs = knowledgeBase.similarity_search(prompt1)

        # Specify the model to be used for generating the summary
        llm = ChatOpenAI(model=MODEL, temperature=0.1)

        # Load a question-answering chain with specified model
        chain = load_qa_chain(llm, chain_type='stuff')

        response = chain.run(input_documents=docs, question=prompt1)
 
    # Use the last summary as the final 2-pager summary
    return response
 
################################################################################################
 
# prompt for 1-pager summary
prompt2 = (
    '''Can you please generate a 1-pager summary (about 250 words only).  also it should contain all the 9 points present in the 
    summary i have provided in paragraphs. The generated text should have proper heading, subheading and content aligned. The heading can include 
    'One Pager Summary for' followed by the company name. \n no need to include conclusion or note in the last. \n try to stick to 
    the word limit.\n Give the summary in paragraphs.'''
)
 
# Function to generate a one-pager summary based on a two-pager summary
def one_pager_summary(two_pager_summary):
    # Process the two-pager summary text to create a knowledge base
    knowledgeBase = process_text(two_pager_summary)

    # Perform similarity search in the knowledge base using prompt3
    docs = knowledgeBase.similarity_search(prompt2)

    # Specify the model to use for generating the summary
    llm = ChatOpenAI(model=MODEL, temperature=0.1)

    # Load a question-answering chain with specified model
    chain = load_qa_chain(llm, chain_type='stuff')

    # Generate the one-pager summary response
    response = chain.run(input_documents=docs, question=prompt2)

    return response

 
# Function to create a download link
def get_download_link(file_path, text):
    with open(file_path, "rb") as file:
        data = file.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{file_path}">{text}</a>'
    return href
 
 
# Create a DOCX file with generated text for two-pager or one-pager summary.
def create_doc(generated_text, heading):
    try:
        with st.spinner(f'Creating DOCX file for {heading}-pager summary...'):
            doc = Document()
            doc.add_paragraph(generated_text)

            # Set narrow margins and justify all paragraphs
            sections = doc.sections
            for section in sections:
                section.top_margin = Pt(36)  # 0.5 inch = 36 points
                section.bottom_margin = Pt(36)
                section.left_margin = Pt(36)
                section.right_margin = Pt(36)
 
            docx_filename = f'{heading}_pager_summary.docx'
            doc.save(docx_filename)
            st.success(f"{heading}-Pager DOCX file created successfully! You can download it below.")
            st.markdown(get_download_link(docx_filename, f"Download {heading}-Pager DOCX file"), unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error creating DOCX file: {e}")
 
 
# Streamlit application for financial report summarization.
def main():
 
    try:
        st.title("Financial Report Summarization")
        
        # Upload the files using streamlit file uploader
        st.sidebar.title("Upload PDF Files")
        uploaded_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True, type="pdf")
        
        if uploaded_files:
            st.sidebar.write(f"You have uploaded {len(uploaded_files)} PDF file(s).")
            st.write("### Uploaded PDF Files:")
            
            # Display the names of uploaded files
            for uploaded_file in uploaded_files:
                st.write(f"**File Name:** {uploaded_file.name}")
            
            # Extract text from each uploaded PDF
 
            texts = [] # Array to store texts generated from each file
            progress_bar_text_extraction = st.progress(0)
    
            with st.spinner('Extracting text from PDFs...'):
                with ThreadPoolExecutor() as executor:
                    # Submit all the tasks to the thread pool
                    futures = [executor.submit(extract_text_from_file, uploaded_file, os.getenv('OPENAI_API_KEY'), idx) 
                            for idx, uploaded_file in enumerate(uploaded_files)]
                    
                    # Collect results as they complete
                    for idx, future in enumerate(as_completed(futures)):
                        texts.append(future.result())
                        progress_bar_text_extraction.progress((idx + 1) / len(uploaded_files))
 
 
 
            # Merge all the extracted texts into a single string
            merged_text = '\n'.join(texts)  
 
            # Generate two-pager summary
            with st.spinner('Generating two-pager summary...'):
                for idx, text in enumerate([merged_text]):
                    two_pager = two_pager_summary([text])
            
            st.text_area("Raw Two-Pager Summary", two_pager, height=300)
            
            # Generate one-pager summary based on two-pager summary
            with st.spinner('Generating one-pager summary...'):
                one_pager = one_pager_summary(two_pager)
            st.text_area("Raw One-Pager Summary", one_pager, height=300)
 
            # Create DOCX file for two-pager summary and display download link
            create_doc(two_pager,"Two")
           
            # Create DOCX file for one-pager summary and display download link
            create_doc(one_pager,"One")
 
 
    except Exception as e:
        st.error(f"An error occurred: {e} ")
 
    
if __name__ == "__main__":
    main()
 