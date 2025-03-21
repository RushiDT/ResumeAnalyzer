import os
import requests
import logging
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"


HUGGINGFACE_API_KEY = "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"

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load NLP Model for similarity
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@app.route('/')
def serve_frontend():
    return send_from_directory("static", "index.html")

@app.route('/upload', methods=['POST'])
def upload_resume():
    if 'file' not in request.files or request.files['file'].filename == '':
        logging.warning("No file uploaded")
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    job_description = request.form.get("job_description", "")
    
    file_ext = os.path.splitext(file.filename)[1]
    if file_ext.lower() not in ['.pdf', '.docx']:
        logging.warning("Unsupported file format")
        return jsonify({"error": "Unsupported file format"}), 400
    
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    if file_ext == ".pdf":
        text = extract_text_from_pdf(file_path)
    else:
        text = extract_text_from_docx(file_path)

    sections = extract_sections(text)
    analysis, match_score = analyze_resume(text, job_description) if job_description else ("No job description provided.", 0)
    
    return jsonify({"analysis": analysis, "match_score": match_score, "sections": sections})

def analyze_resume(resume_text, job_description):
    if not HUGGINGFACE_API_KEY:
        logging.error("Hugging Face API key is missing")
        return "Error: API key is missing. Please check your environment variables.", 0
    
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"question": "What are the key strengths of this resume?", "context": resume_text}

        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload)
        response.raise_for_status()

        api_response = response.json()
        logging.info(f"Hugging Face API response: {api_response}")  # Debug log

        analysis = api_response.get("answer", "No insights generated.") if isinstance(api_response, dict) else "No insights generated."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Hugging Face API: {e}")
        analysis = f"Error: {str(e)}"

    # Calculate similarity score
    resume_embedding = similarity_model.encode(resume_text, convert_to_tensor=True)
    job_embedding = similarity_model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item() * 100

    return analysis, round(similarity_score, 2)

def extract_text_from_pdf(file_path):
    import pdfplumber
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()

def extract_text_from_docx(file_path):
    from docx import Document
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_sections(text):
    return text.split("\n\n")

if __name__ == '__main__':
    app.run(debug=True)
