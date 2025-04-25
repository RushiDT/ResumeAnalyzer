import os
import requests
import logging
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load environment variables
load_dotenv()
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"

# Add a configuration variable to disable the API call in case of issues
USE_HUGGINGFACE_API = False  # Set to True when you have a valid API key
DUMMY_ANALYSIS = "Based on the resume analysis, the candidate shows skills in [relevant area]. The resume highlights experience in [field] with strengths in [skills]. Consider focusing more on quantifiable achievements and tailoring specific skills to the job requirements."

# Get API key from environment or use the hardcoded one (not recommended for production)
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize vectorizer for text similarity
vectorizer = TfidfVectorizer()

@app.route('/')
def serve_frontend():
    return send_from_directory("static", "index.html")

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory("static", path)

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

    try:
        if file_ext.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)
        else:
            text = extract_text_from_docx(file_path)

        sections = extract_sections(text)
        analysis, match_score = analyze_resume(text, job_description) if job_description else ("No job description provided.", 0)
        
        return jsonify({"analysis": analysis, "match_score": match_score, "sections": sections})
    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

def analyze_resume(resume_text, job_description):
    # Calculate similarity score using TF-IDF
    try:
        # Ensure we have enough content to vectorize
        if len(resume_text.split()) < 5 or len(job_description.split()) < 5:
            similarity_score = 0
        else:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_description])
            similarity_score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100
    except Exception as e:
        logging.error(f"Error calculating similarity: {e}")
        similarity_score = 0

    # If we're not using the HuggingFace API, return a dummy analysis
    if not USE_HUGGINGFACE_API:
        # Extract key terms from job description to make the analysis more relevant
        job_terms = extract_key_terms(job_description)
        analysis = DUMMY_ANALYSIS
        for term in job_terms[:3]:  # Use up to 3 key terms
            if term:
                analysis = analysis.replace("[relevant area]", term, 1)
                analysis = analysis.replace("[field]", term, 1)
                analysis = analysis.replace("[skills]", term, 1)
        
        # Replace any remaining placeholders
        analysis = analysis.replace("[relevant area]", "software development")
        analysis = analysis.replace("[field]", "technology")
        analysis = analysis.replace("[skills]", "technical and soft skills")
        
        return analysis, round(similarity_score, 2)

    # Otherwise, try to use the HuggingFace API
    if not HUGGINGFACE_API_KEY:
        logging.error("Hugging Face API key is missing")
        return "Error: API key is missing. Please check your environment variables.", round(similarity_score, 2)
    
    try:
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}
        payload = {"question": "What are the key strengths of this resume?", "context": resume_text}

        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=payload, timeout=10)
        response.raise_for_status()

        api_response = response.json()
        logging.info(f"Hugging Face API response: {api_response}")  # Debug log

        analysis = api_response.get("answer", "No insights generated.") if isinstance(api_response, dict) else "No insights generated."
    except requests.exceptions.RequestException as e:
        logging.error(f"Error calling Hugging Face API: {e}")
        analysis = f"Error connecting to analysis service. Using similarity score only."
        # Fall back to the dummy analysis in case of API failure
        return DUMMY_ANALYSIS, round(similarity_score, 2)

    return analysis, round(similarity_score, 2)

def extract_key_terms(text):
    """Extract potential key terms from the job description"""
    # Split by common delimiters and get unique terms
    words = text.lower().replace('\n', ' ').replace(',', ' ').replace('.', ' ').split()
    # Filter out common words and short words
    stopwords = {'the', 'and', 'a', 'to', 'of', 'in', 'with', 'for', 'on', 'at', 'from', 'by', 'about', 'as', 'an', 'is', 'are', 'be', 'will', 'you', 'your', 'we', 'our'}
    terms = [word for word in words if word not in stopwords and len(word) > 3]
    # Return most frequent terms
    term_freq = {}
    for term in terms:
        term_freq[term] = term_freq.get(term, 0) + 1
    # Sort by frequency
    sorted_terms = sorted(term_freq.items(), key=lambda x: x[1], reverse=True)
    return [term for term, freq in sorted_terms]

def extract_text_from_pdf(file_path):
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        return text.strip()
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        raise Exception(f"Could not extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path):
    try:
        from docx import Document
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
    except Exception as e:
        logging.error(f"Error extracting text from DOCX: {e}")
        raise Exception(f"Could not extract text from DOCX: {str(e)}")

def extract_sections(text):
    """Extract meaningful sections from the resume text"""
    # Simple heuristic: split by double newlines and filter empty sections
    sections = [section.strip() for section in text.split("\n\n")]
    return [section for section in sections if section]

if __name__ == '__main__':
    app.run(debug=True)