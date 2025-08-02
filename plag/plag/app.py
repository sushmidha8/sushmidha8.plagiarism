from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
import numpy as np
import os
from werkzeug.utils import secure_filename
import PyPDF2
import docx
from collections import defaultdict
import traceback
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import jellyfish

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'doc', 'docx'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Simulated user database
users = {
    "admin@example.com": "securepassword123"
}

# Load BERT model with better error handling
print("Initializing BERT model...")
try:
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = TFBertModel.from_pretrained('bert-base-uncased')
    print("✅ BERT model loaded successfully!")
except Exception as e:
    print(f"❌ Failed to load BERT model: {str(e)}")
    tokenizer = None
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_file(file_path):
    try:
        ext = file_path.split('.')[-1].lower()
        if ext == "txt":
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        elif ext == "pdf":
            text = ""
            with open(file_path, "rb") as f:
                pdf_reader = PyPDF2.PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text() + " "
            return text.strip()
        elif ext in {"doc", "docx"}:
            doc = docx.Document(file_path)
            return " ".join([para.text.strip() for para in doc.paragraphs]).strip()
        return ""
    except Exception as e:
        print(f"Error extracting text from file: {str(e)}")
        return ""

def get_bert_embedding(text):
    if not tokenizer or not model:
        raise ValueError("BERT model or tokenizer not loaded")
    
    try:
        inputs = tokenizer(text, return_tensors='tf', padding=True, truncation=True, max_length=512)
        outputs = model(**inputs)
        
        # Use mean pooling with attention mask
        token_embeddings = outputs.last_hidden_state
        input_mask = tf.cast(inputs['attention_mask'], tf.float32)
        input_mask = tf.expand_dims(input_mask, axis=-1)
        
        sum_embeddings = tf.reduce_sum(token_embeddings * input_mask, axis=1)
        sum_mask = tf.reduce_sum(input_mask, axis=1)
        mean_embeddings = sum_embeddings / sum_mask
        
        # Normalize embeddings
        normalized_embedding = tf.math.l2_normalize(mean_embeddings, axis=1)
        return normalized_embedding
    except Exception as e:
        print(f"Error generating BERT embeddings: {str(e)}")
        raise

def calculate_similarity(text1, text2):
    try:
        if not text1.strip() or not text2.strip():
            return 0.0
        
        # Calculate multiple similarity metrics
        scores = {
            'bert': calculate_bert_similarity(text1, text2),
            'tfidf': calculate_tfidf_similarity(text1, text2),
            'jaccard': calculate_jaccard_similarity(text1, text2),
            'levenshtein': calculate_levenshtein_similarity(text1, text2)
        }
        
        # Weighted average based on expected accuracy
        final_score = (
            0.5 * scores['bert'] + 
            0.3 * scores['tfidf'] + 
            0.1 * scores['jaccard'] + 
            0.1 * scores['levenshtein']
        )
        
        return max(0.0, min(1.0, final_score))  # Ensure between 0 and 1
    
    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

def calculate_bert_similarity(text1, text2):
    """Calculate similarity using BERT embeddings"""
    if not tokenizer or not model:
        return 0.0
    
    try:
        embedding1 = get_bert_embedding(text1)
        embedding2 = get_bert_embedding(text2)
        similarity = tf.reduce_sum(tf.multiply(embedding1, embedding2), axis=1)
        return float(similarity.numpy()[0])
    except:
        return 0.0

def cosine_similarity_score(embedding1, embedding2):
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def calculate_tfidf_similarity(text1, text2):
    """Calculate similarity using TF-IDF vectors"""
    try:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        return 0.0

def calculate_jaccard_similarity(text1, text2):
    """Calculate Jaccard similarity between texts"""
    try:
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def calculate_levenshtein_similarity(text1, text2):
    """Calculate normalized Levenshtein similarity"""
    try:
        distance = jellyfish.levenshtein_distance(text1[:100], text2[:100])  # Limit to first 100 chars
        max_len = max(len(text1[:100]), len(text2[:100]))
        return 1 - (distance / max_len) if max_len > 0 else 0.0
    except:
        return 0.0


# ==================== DATA STRUCTURES ====================

class BPlusTreeNode:
    def __init__(self, leaf=False):
        self.keys = []
        self.values = []
        self.leaf = leaf
        self.children = []
        self.next = None

class BPlusTree:
    def __init__(self, order=3):
        self.root = BPlusTreeNode(leaf=True)
        self.order = order
    
    def insert(self, key, value):
        if not self.root:
            self.root = BPlusTreeNode(leaf=True)
        self._insert(self.root, key, value)
    
    def _insert(self, node, key, value):
        if node.leaf:
            self._insert_leaf(node, key, value)
        else:
            self._insert_non_leaf(node, key, value)
    
    def _insert_leaf(self, node, key, value):
        node.keys.append(key)
        node.values.append(value)
        node.keys, node.values = zip(*sorted(zip(node.keys, node.values)))
        node.keys, node.values = list(node.keys), list(node.values)

class SplayNode:
    def __init__(self, key, value):
        self.key = key
        self.value = value
        self.left = None
        self.right = None

class SplayTree:
    def __init__(self):
        self.root = None
    
    def insert(self, key, value):
        if not self.root:
            self.root = SplayNode(key, value)
            return
        self.root = self._splay(self.root, key)
        if key < self.root.key:
            node = SplayNode(key, value)
            node.left = self.root.left
            node.right = self.root
            self.root.left = None
            self.root = node
        elif key > self.root.key:
            node = SplayNode(key, value)
            node.right = self.root.right
            node.left = self.root
            self.root.right = None
            self.root = node

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
    
    def search(self, word):
        node = self.root
        for char in word.lower():
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

# Initialize data structures (they won't interfere with existing code)
document_bplus_tree = BPlusTree(order=4)
document_splay_tree = SplayTree()
phrase_trie = Trie()

# Routes
@app.route('/')
def home():
    return redirect(url_for('login_page'))

@app.route('/login', methods=['GET', 'POST'])
def login_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        if email in users and users[email] == password:
            return redirect(url_for('check_page'))
        flash('Invalid credentials', 'error')
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup_page():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        if password != confirm_password:
            flash('Passwords do not match', 'error')
        elif email in users:
            flash('Email already registered', 'error')
        else:
            users[email] = password
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login_page'))
    return render_template('signin.html')

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password_page():
    if request.method == 'POST':
        email = request.form.get('email')
        if email in users:
            return redirect(url_for('reset_password_page', email=email))
        else:
            flash('Email not registered', 'error')
    return render_template('forgot-password.html')

@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password_page():
    if request.method == 'POST':
        email = request.form.get('email')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if email not in users:
            flash('Email not registered', 'error')
        elif new_password != confirm_password:
            flash('Passwords do not match', 'error')
        else:
            users[email] = new_password
            flash('Password reset successfully! Please log in.', 'success')
            return redirect(url_for('login_page'))
    
    email = request.args.get('email')
    return render_template('reset-password.html', email=email)

@app.route('/check')
def check_page():
    return render_template('check.html')

@app.route('/model-status')
def model_status():
    return jsonify({
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

@app.route('/test-similarity')
def test_similarity():
    # Test cases with expected ranges
    test_cases = [
        {
            "text1": "The quick brown fox jumps over the lazy dog",
            "text2": "The quick brown fox jumps over the lazy dog",
            "expected_min": 0.95,
            "expected_max": 1.0
        },
        {
            "text1": "The quick brown fox jumps over the lazy dog",
            "text2": "A fast brown fox leaps over a sleepy dog",
            "expected_min": 0.7,
            "expected_max": 0.9
        },
        {
            "text1": "Machine learning is fascinating",
            "text2": "Artificial intelligence is interesting",
            "expected_min": 0.4,
            "expected_max": 0.6
        },
        {
            "text1": "This is completely different",
            "text2": "No relation whatsoever",
            "expected_min": 0.0,
            "expected_max": 0.2
        }
    ]
    
    results = []
    for case in test_cases:
        score = calculate_similarity(case['text1'], case['text2'])
        results.append({
            "text1": case['text1'],
            "text2": case['text2'],
            "score": score,
            "expected_range": f"{case['expected_min']}-{case['expected_max']}",
            "passed": case['expected_min'] <= score <= case['expected_max']
        })
    
    return jsonify({"results": results})

@app.route('/check-similarity', methods=['POST'])
def check_similarity():
    try:
        # Initialize variables
        doc1_text = request.form.get('doc1', '').strip()
        doc2_text = request.form.get('doc2', '').strip()
        
        # Process file uploads
        for file_key in ['file1', 'file2']:
            if file_key in request.files:
                file = request.files[file_key]
                if file and file.filename != '' and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file.save(filepath)
                    extracted = extract_text_from_file(filepath)
                    if file_key == 'file1':
                        doc1_text += " " + extracted
                    else:
                        doc2_text += " " + extracted
                    os.remove(filepath)
        
        # Validate input
        doc1_text = doc1_text.strip()
        doc2_text = doc2_text.strip()
        
        if not doc1_text or not doc2_text:
            return jsonify({"error": "Both documents must contain text"}), 400
        
        # Calculate similarity
        similarity_score = calculate_similarity(doc1_text, doc2_text)
        
        # Get common words
        common_words = list(set(doc1_text.lower().split()) & set(doc2_text.lower().split()))
        common_words = [w for w in common_words if len(w) > 3][:10]  # Filter short words
        
        # Prepare response
        response = {
            "similarity_score": similarity_score,
            "doc1_length": len(doc1_text),
            "doc2_length": len(doc2_text),
            "common_words": common_words,
            "interpretation": get_interpretation(similarity_score)
        }
        
        return jsonify(response)
        
    except Exception as e:
        app.logger.error(f"Error in check_similarity: {str(e)}\n{traceback.format_exc()}")
        return jsonify({
            "error": "An error occurred while processing your request",
            "details": str(e)
        }), 500

def get_interpretation(score):
    """Provide human-readable interpretation of similarity score"""
    if score >= 0.9:
        return "Very high similarity - documents are nearly identical"
    elif score >= 0.7:
        return "High similarity - documents convey similar meaning with some variation"
    elif score >= 0.5:
        return "Moderate similarity - some shared concepts but significant differences"
    elif score >= 0.3:
        return "Low similarity - minimal shared content"
    else:
        return "Very low similarity - documents appear unrelated"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)