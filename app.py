import streamlit as st
import requests
import os
from streamlit_lottie import st_lottie
from dotenv import load_dotenv
import google.generativeai as genai
import openai
import fitz  # PyMuPDF for reading PDFs
import pytesseract  # For OCR on scanned PDFs
from PIL import Image
import io
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from bs4 import BeautifulSoup
import trafilatura
import urllib.parse
import validators
from hashlib import md5
import json
import random
from typing import List, Dict

# === FIRST COMMAND ===
st.set_page_config(page_title="Student Assistant Chatbot", page_icon="🤖", layout="wide")

# === Setup Tesseract path ===
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# === Load environment variables ===
load_dotenv()



# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # Get key from .env
if not GEMINI_API_KEY:
    raise ValueError("❌ Gemini API key not found in .env file")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Get key from .env
if not OPENAI_API_KEY:
    raise ValueError("❌ Gemini API key not found in .env file")


genai.configure(api_key=GEMINI_API_KEY)


if not GEMINI_API_KEY or not OPENAI_API_KEY:
    st.error("❌ Error: Missing API keys. Please check your .env file.")
    st.stop()

# Configure APIs
genai.configure(api_key=GEMINI_API_KEY)
openai.api_key = OPENAI_API_KEY

# === Web Content Cache Setup ===
WEB_CACHE_DIR = "web_cache"
os.makedirs(WEB_CACHE_DIR, exist_ok=True)

def get_cache_key(url):
    return md5(url.encode()).hexdigest()

def cache_web_content(url, content):
    with open(os.path.join(WEB_CACHE_DIR, get_cache_key(url)), "w") as f:
        json.dump({"url": url, "content": content}, f)

def get_cached_content(url):
    try:
        with open(os.path.join(WEB_CACHE_DIR, get_cache_key(url))) as f:
            return json.load(f)["content"]
    except:
        return None

# === Apply Custom CSS ===
def apply_custom_css():
    st.markdown("""
        <style>
            body { background-color: #f0f2f6; color: #333333; }
            .main { background-color: #ffffff; border-radius: 10px; padding: 20px; }
            .stButton button {
                background-color: #4CAF50; color: white; border-radius: 8px;
                padding: 10px 20px; border: none; cursor: pointer; font-size: 16px;
            }
            .stButton button:hover { background-color: #45a049; }
            .chat-bubble {
                background-color: #e1f5fe; border-radius: 10px; padding: 10px; margin: 5px 0;
            }
            .tab-content { padding: 15px; border-radius: 8px; background-color: #f9f9f9; }
            .quiz-question { 
                background-color: #f8f9fa; 
                padding: 15px; 
                border-radius: 8px; 
                margin-bottom: 15px;
            }
            .quiz-option {
                padding: 8px;
                margin: 5px 0;
            }
            .quiz-correct {
                background-color: #d4edda;
            }
            .quiz-incorrect {
                background-color: #f8d7da;
            }
        </style>
    """, unsafe_allow_html=True)

apply_custom_css()

# === Add Lottie Animation ===
def load_lottie_url(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_animation = load_lottie_url("https://assets7.lottiefiles.com/packages/lf20_jcikwtux.json")

# === Initialize session state ===
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'images_data' not in st.session_state:
    st.session_state.images_data = []
if 'quiz_questions' not in st.session_state:
    st.session_state.quiz_questions = []
if 'quiz_score' not in st.session_state:
    st.session_state.quiz_score = 0
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'quiz_answers' not in st.session_state:
    st.session_state.quiz_answers = {}

# === Document Processing Functions ===
def ocr_page(page):
    pix = page.get_pixmap()
    img_bytes = pix.tobytes("png")
    image = Image.open(io.BytesIO(img_bytes))
    text = pytesseract.image_to_string(image)
    return text

def is_scanned_pdf(doc):
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        if page.get_text().strip():
            return False
    return True

def extract_images_and_captions(doc):
    images_data = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        images = page.get_images(full=True)
        
        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            pil_img = Image.open(io.BytesIO(image_bytes))

            caption = pytesseract.image_to_string(pil_img).strip()
            if not caption:
                caption = f"Diagram on page {page_num + 1}, image {img_index + 1}"

            embedding = np.array(pil_img.convert("L")).flatten().mean()

            images_data.append({
                "page": page_num + 1,
                "image_index": img_index + 1,
                "caption": caption,
                "image": pil_img,
                "embedding": np.array([[embedding]])
            })
    return images_data

def load_docs(uploaded_files):
    documents = []
    all_images = []
    
    for uploaded_file in uploaded_files:
        st.info(f"📄 Processing: {uploaded_file.name}")
        try:
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")

            if is_scanned_pdf(doc):
                st.warning(f"🧐 Scanned PDF detected: {uploaded_file.name}. Running OCR...")
                content = ""
                for page_num in range(len(doc)):
                    page_text = ocr_page(doc.load_page(page_num))
                    content += f"\n\nPage {page_num + 1}:\n{page_text}"
            else:
                content = ""
                for page_num in range(len(doc)):
                    page_text = doc.load_page(page_num).get_text().strip()
                    content += f"\n\nPage {page_num + 1}:\n{page_text}"

            documents.append({
                "name": uploaded_file.name,
                "page_content": content,
                "source_type": "pdf"
            })

            images = extract_images_and_captions(doc)
            all_images.extend(images)

        except Exception as e:
            st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")

    st.session_state.images_data = all_images
    return documents

def process_webpage(url):
    try:
        cached_content = get_cached_content(url)
        if cached_content:
            return {
                "name": f"Webpage: {url}",
                "page_content": cached_content,
                "source_type": "webpage"
            }, None
            
        if not validators.url(url):
            return None, "❌ Invalid URL format"
            
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return None, "❌ Failed to fetch URL content"
            
        content = trafilatura.extract(downloaded, include_comments=False)
        if not content:
            return None, "❌ No extractable content found"
        
        soup = BeautifulSoup(content, 'html.parser')
        for script in soup(["script", "iframe", "object"]):
            script.decompose()
        clean_content = str(soup)
        
        cache_web_content(url, clean_content)
            
        return {
            "name": f"Webpage: {url}",
            "page_content": clean_content,
            "source_type": "webpage"
        }, None
        
    except Exception as e:
        return None, f"❌ Error processing webpage: {str(e)}"

# === AI Functions ===
def find_most_relevant_documents(question, documents, top_n=4):
    if not documents:
        return []

    contents = [doc["page_content"] for doc in documents]
    vectorizer = TfidfVectorizer().fit_transform([question] + contents)
    similarity_scores = cosine_similarity(vectorizer[0:1], vectorizer[1:]).flatten()

    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    return [documents[idx] for idx in top_indices]

def search_images(query, images_data):
    if not images_data:
        return []

    query_embedding = np.array([[len(query)]])
    results = []

    for img_info in images_data:
        similarity = cosine_similarity(query_embedding, img_info["embedding"])[0][0]
        if similarity > 0.7:
            results.append({
                "image": img_info["image"],
                "caption": img_info["caption"],
                "similarity": similarity
            })

    return sorted(results, key=lambda x: x["similarity"], reverse=True)

def ask_gemini(question, documents):
    if not documents:
        return "❌ No documents found. Please upload PDFs or add web content first."

    most_relevant_docs = find_most_relevant_documents(question, documents)
    if not most_relevant_docs:
        return "❌ Couldn't find relevant content in the documents."

    context = "\n\n*".join([doc["page_content"] for doc in most_relevant_docs])
    prompt = f"Use the following documents to answer:\n\n{context}\n\nQuestion: {question}"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        response_text = response.text if response else "🤖 No answer generated."

        source_names = [doc["name"] for doc in most_relevant_docs]
        source_text = "\n\n📚 Sources:\n- " + "\n- ".join(source_names)
        
        bot_response = response_text + source_text

        keywords = ["image", "diagram", "chart", "illustration", "figure"]
        if any(k in question.lower() for k in keywords):
            images_data = st.session_state.images_data
            search_results = search_images(question, images_data)
            
            if search_results:
                best_match = search_results[0]
                resized_img = best_match["image"].resize((400, int(400 * best_match["image"].height / best_match["image"].width)))
                
                st.markdown(f"<div class='chat-bubble'><strong>Bot:</strong> {bot_response}</div>", unsafe_allow_html=True)
                st.markdown("🖼 *Relevant image found:*")
                st.image(resized_img, caption=best_match["caption"], use_container_width=True)
                return ""
            else:
                return bot_response + "\n\n⚠ No matching image found."

        return bot_response

    except Exception as e:
        return f"❌ Gemini error: {str(e)}"

def extract_key_insights(documents):
    if not documents:
        return "❌ No documents found."

    context = "\n\n*".join([doc["page_content"] for doc in documents])
    prompt = f"Extract key insights from these documents:\n\n{context}"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else "🤖 No insights generated."
    except Exception as e:
        return f"❌ Error: {str(e)}"

def summarize_document(documents):
    if not documents:
        return "❌ No documents found."

    context = "\n\n*".join([doc["page_content"] for doc in documents])
    prompt = f"Summarize these documents:\n\n{context}"

    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text if response else "🤖 No summary generated."
    except Exception as e:
        return f"❌ Error: {str(e)}"

# === Quiz Functions ===
def generate_quiz_questions(documents: List[Dict], num_questions: int = 5) -> List[Dict]:
    """
    Generate MCQ quiz questions from document content using Gemini
    Returns a list of questions with answers and options
    """
    if not documents:
        return []

    # Combine relevant content
    context = "\n\n".join([doc["page_content"][:10000] for doc in documents])
    
    prompt = f"""Generate exactly {num_questions} multiple choice quiz questions based on this content:
    {context}
    
    For each question:
    1. Make it clear and specific
    2. Provide exactly 4 answer options (labeled a-d)
    3. The 'answer' field must contain the FULL correct option text
    4. Format as JSON with:
    {{
        "question": "text",
        "options": ["a) Option 1", "b) Option 2", "c) Option 3", "d) Option 4"],
        "answer": "b) Option 2"  // MUST be the full option text
    }}
    Return ONLY the JSON array with no additional text.
    """
    
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        
        if response.text:
            try:
                # Clean response to extract JSON
                json_str = response.text.replace("json", "").replace("", "").strip()
                questions = json.loads(json_str)
                
                # Validate each question
                for q in questions:
                    if not all(k in q for k in ["question", "options", "answer"]):
                        raise ValueError("Missing required fields")
                    if q["answer"] not in q["options"]:
                        raise ValueError(f"Answer '{q['answer']}' not in options")
                
                return questions[:num_questions]
                
            except (json.JSONDecodeError, ValueError) as e:
                st.error(f"Error parsing questions: {str(e)}")
                st.text("Raw response:")
                st.text(response.text)
                return []
            
    except Exception as e:
        st.error(f"Quiz generation error: {str(e)}")
        return []    
def display_quiz():
    if not st.session_state.quiz_questions:
        st.warning("No questions generated")
        return
        
    st.subheader("📝 Quiz Time!")
    
    # Initialize session state variables
    if 'user_answers' not in st.session_state:
        st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
    if 'quiz_submitted' not in st.session_state:
        st.session_state.quiz_submitted = False
    if 'score' not in st.session_state:
        st.session_state.score = 0
    
    with st.form("quiz_form"):
        for i, question in enumerate(st.session_state.quiz_questions):
            st.markdown(f"{i+1}. {question['question']}")
            
            # Display radio buttons for options
            selected_option = st.radio(
                "Select your answer:",
                question["options"],
                key=f"question_{i}",
                index=None
            )
            
            # Store the selected option
            if not st.session_state.quiz_submitted:
                st.session_state.user_answers[i] = selected_option
            
            # Show feedback after submission
            if st.session_state.quiz_submitted:
                if st.session_state.user_answers[i] == question["answer"]:
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Incorrect. The correct answer is: {question['answer']}")
        
        # Submit or Try Again button
        if not st.session_state.quiz_submitted:
            if st.form_submit_button("Submit Quiz"):
                st.session_state.quiz_submitted = True
                # Calculate score
                st.session_state.score = sum(
                    1 for i, q in enumerate(st.session_state.quiz_questions)
                    if st.session_state.user_answers[i] == q["answer"]
                )
                st.rerun()
        else:
            st.success(f"Your score: {st.session_state.score}/{len(st.session_state.quiz_questions)}")
            if st.form_submit_button("Try Again"):
                # Reset quiz state
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = [None] * len(st.session_state.quiz_questions)
                st.session_state.score = 0
                st.rerun()    
# === MAIN APPLICATION ===
with st.container():
    left, right = st.columns([2, 1])
    with left:
        st.title("🤖 Student Assistant Chatbot")
        st.subheader("Your Smart Study Companion")
        st.write("Upload documents, add web content, ask questions, get summaries, insights, and quizzes!")
    with right:
        if lottie_animation:
            st_lottie(lottie_animation, height=200, key="chatbot")

# Sidebar
with st.sidebar:
    st.header("📂 Upload Content")
    
    tab1, tab2 = st.tabs(["PDF Upload", "Web Content"])
    
    with tab1:
        uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        if uploaded_files:
            with st.spinner("Processing files..."):
                docs = load_docs(uploaded_files)
                st.session_state.documents.extend(docs)
            st.success(f"✅ Loaded {len(docs)} PDF document(s).")
    
    with tab2:
        url_input = st.text_input("Enter webpage URL:")
        
        if st.button("Add Webpage"):
            if url_input:
                with st.spinner("Processing webpage..."):
                    web_doc, error = process_webpage(url_input)
                    if web_doc:
                        st.session_state.documents.append(web_doc)
                        st.success(f"✅ Added webpage content")
                    else:
                        st.error(error)
            else:
                st.warning("Please enter a URL")

    st.markdown("---")

    if st.button("📄 Summarize Documents"):
        if st.session_state.documents:
            with st.spinner("Generating summary..."):
                summary = summarize_document(st.session_state.documents)
                st.session_state.chat_history.append(("Summarize Documents", summary))
        else:
            st.warning("❌ Please upload documents or add web content.")

    if st.button("💡 Extract Key Insights"):
        if st.session_state.documents:
            with st.spinner("Extracting insights..."):
                insights = extract_key_insights(st.session_state.documents)
                st.session_state.chat_history.append(("Key Insights", insights))
        else:
            st.warning("❌ Please upload documents or add web content.")
    
    st.markdown("---")
    
    if st.button("📝 Generate Quiz", help="Create practice questions from your documents"):
        if st.session_state.documents:
            with st.spinner("Creating quiz questions..."):
                questions = generate_quiz_questions(
                    st.session_state.documents,
                    num_questions=5
                )
                
                if questions:
                    st.session_state.quiz_questions = questions
                    st.session_state.quiz_submitted = False
                    st.session_state.user_answers = {}
                    st.session_state.score = 0
                    st.success(f"✅ Generated {len(questions)} quiz questions!")
                    st.balloons()
                else:
                    st.error("❌ Failed to generate quiz. Please try again.")
        else:
            st.warning("⚠ Please upload documents first")

# Chat Section
st.subheader("💬 Ask questions about your content")
user_input = st.chat_input("Type your question here...")
if user_input:
    with st.spinner("🤖 Thinking..."):
        bot_reply = ask_gemini(user_input, st.session_state.documents)
        st.session_state.chat_history.append((user_input, bot_reply))

# Chat History Display
for user_msg, bot_msg in st.session_state.chat_history:
    st.markdown(f"<div class='chat-bubble'><strong>You:</strong> {user_msg}</div>", unsafe_allow_html=True)
    if bot_msg:
        st.markdown(f"<div class='chat-bubble'><strong>Bot:</strong> {bot_msg}</div>", unsafe_allow_html=True)

# Quiz Display
if st.session_state.quiz_questions:
    display_quiz()