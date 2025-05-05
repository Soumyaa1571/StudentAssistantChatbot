
# import os
# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# import streamlit as st
# from tempfile import NamedTemporaryFile

# @st.cache_data(ttl=300)  # Cache for 5 minutes
# def load_docs(uploaded_files=[]):
#     documents = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)  # Split into 1000-character chunks

#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             try:
#                 with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
#                     temp_file.write(uploaded_file.getvalue())
#                     temp_file_path = temp_file.name

#                 if uploaded_file.name.endswith('.pdf'):
#                     print(f"hello")
#                     loader = PyPDFLoader(temp_file_path)
#                 elif uploaded_file.name.endswith('.docx') or uploaded_file.name.endswith('.doc'):
#                     loader = Docx2txtLoader(temp_file_path)
#                 elif uploaded_file.name.endswith('.txt'):
#                     loader = TextLoader(temp_file_path)
#                 else:
#                     continue  # Skip unsupported file types

#                 loaded_docs = loader.load()
#                 print(f"content of loaded_docs",loaded_docs)
#                 if loaded_docs:
#                     print(f"yes loaded_docs")
#                     split_docs = text_splitter.split_documents(loaded_docs)  # Split text into chunks
#                     documents.extend(split_docs)
#                     print(f"✅ Loaded {uploaded_file.name} into {len(split_docs)} chunks.")

#                     for i, doc in enumerate(split_docs):
#                         print(f"📄 Document {i + 1} (first 500 chars): {doc.page_content[:500]}")


#             except Exception as e:
#                 print(f"❌ Failed to load {uploaded_file.name}: {e}")

#             finally:
#                 os.remove(temp_file_path)

#     print(f"final documents is",documents)
#     return documents if documents else []


# from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
# from pdf2image import convert_from_path

# import pytesseract
# from langchain.schema import Document  # Optional for structured docs

# # Path to tesseract.exe if not set globally
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# @st.cache_data(ttl=300)
# def load_docs(uploaded_files=[]):
#     documents = []
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

#     if uploaded_files:
#         for uploaded_file in uploaded_files:
#             try:
#                 with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
#                     temp_file.write(uploaded_file.getvalue())
#                     temp_file_path = temp_file.name

#                 if uploaded_file.name.endswith('.pdf'):
#                     print("hello")
#                     loader = PyPDFLoader(temp_file_path)
#                     loaded_docs = loader.load()

#                     if not loaded_docs:  # If no text found, try OCR
#                         print("No text found in PDF. Running OCR...")
#                         images = convert_from_path(temp_file_path)
#                         text = ''
#                         for img in images:
#                             text += pytesseract.image_to_string(img)

#                         loaded_docs = [Document(page_content=text)]

#                 elif uploaded_file.name.endswith('.docx') or uploaded_file.name.endswith('.doc'):
#                     loader = Docx2txtLoader(temp_file_path)
#                     loaded_docs = loader.load()

#                 elif uploaded_file.name.endswith('.txt'):
#                     loader = TextLoader(temp_file_path)
#                     loaded_docs = loader.load()

#                 else:
#                     continue  # Skip unsupported file types

#                 if loaded_docs:
#                     split_docs = text_splitter.split_documents(loaded_docs)
#                     documents.extend(split_docs)
#                     print(f"✅ Loaded {uploaded_file.name} into {len(split_docs)} chunks.")

#                     for i, doc in enumerate(split_docs):
#                         print(f"📄 Document {i + 1} (first 500 chars): {doc.page_content[:500]}")

#             except Exception as e:
#                 print(f"❌ Failed to load {uploaded_file.name}: {e}")

#             finally:
#                 os.remove(temp_file_path)

#     print(f"Final documents: {documents}")
#     return documents if documents else []


from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document  # For structured documents
from tempfile import NamedTemporaryFile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pytesseract
import os
import fitz  # PyMuPDF for image extraction
from PIL import Image
import io

# Ensure Tesseract is properly linked
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# === Cache Documents Loading ===
@st.cache_data(ttl=300)
def load_docs(uploaded_files=[]):
    documents = []
    all_images = []  # To store extracted images with captions

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    if uploaded_files:
        for uploaded_file in uploaded_files:
            try:
                with NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                    temp_file.write(uploaded_file.getvalue())
                    temp_file_path = temp_file.name

                file_ext = os.path.splitext(uploaded_file.name)[1].lower()

                loaded_docs = []
                # === PDF Handling ===
                if file_ext == '.pdf':
                    st.info(f"📄 Processing PDF: {uploaded_file.name}")
                    loader = PyPDFLoader(temp_file_path)
                    loaded_docs = loader.load()

                    if not loaded_docs:
                        st.warning("🧐 No text found in PDF. Running OCR...")
                        images = convert_from_path(temp_file_path)
                        ocr_text = ""
                        for img in images:
                            ocr_text += pytesseract.image_to_string(img)
                        loaded_docs = [Document(page_content=ocr_text)]

                    # === Extract images from PDF ===
                    doc = fitz.open(temp_file_path)
                    extracted_images = extract_images_and_captions(doc)
                    all_images.extend(extracted_images)

                # === DOCX Handling ===
                elif file_ext in ['.doc', '.docx']:
                    st.info(f"📄 Processing Word Document: {uploaded_file.name}")
                    loader = Docx2txtLoader(temp_file_path)
                    loaded_docs = loader.load()

                # === TXT Handling ===
                elif file_ext == '.txt':
                    st.info(f"📄 Processing Text File: {uploaded_file.name}")
                    loader = TextLoader(temp_file_path)
                    loaded_docs = loader.load()

                else:
                    st.warning(f"⚠️ Unsupported file type: {file_ext}")
                    continue  # Skip unsupported files

                # === Split into Chunks ===
                if loaded_docs:
                    split_docs = text_splitter.split_documents(loaded_docs)
                    documents.extend(split_docs)
                    st.success(f"✅ Loaded {uploaded_file.name} into {len(split_docs)} chunks.")

                    for i, doc in enumerate(split_docs):
                        print(f"📄 Chunk {i + 1} (first 300 chars): {doc.page_content[:300]}...")

            except Exception as e:
                st.error(f"❌ Failed to load {uploaded_file.name}: {e}")

            finally:
                os.remove(temp_file_path)

    # Save all extracted images globally for querying later
    st.session_state['images_data'] = all_images
    print(f"✅ Total documents processed: {len(documents)}")
    print(f"✅ Total images extracted: {len(all_images)}")

    return documents if documents else []
