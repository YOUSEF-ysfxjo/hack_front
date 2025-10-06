from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
from family_system import FamilyMatcher
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path

# Initialize Flask app
app = Flask(__name__, static_folder='frontend/build', static_url_path='')
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the family matcher
matcher = FamilyMatcher()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API Routes
@app.route('/api/upload', methods=['POST'])
def upload_files():
    if 'mother_img' not in request.files or 'father_img' not in request.files:
        return jsonify({'error': 'Both mother and father images are required'}), 400
    
    mother_file = request.files['mother_img']
    father_file = request.files['father_img']
    
    if mother_file.filename == '' or father_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not (mother_file and father_file and 
            allowed_file(mother_file.filename) and 
            allowed_file(father_file.filename)):
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400
    
    try:
        # Save uploaded files with timestamps
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        mother_filename = f'mother_{timestamp}_{secure_filename(mother_file.filename)}'
        father_filename = f'father_{timestamp}_{secure_filename(father_file.filename)}'
        
        mother_path = os.path.join(app.config['UPLOAD_FOLDER'], mother_filename)
        father_path = os.path.join(app.config['UPLOAD_FOLDER'], father_filename)
        
        mother_file.save(mother_path)
        father_file.save(father_path)
        
        # Process images and get results
        try:
            # Find the best alpha value (using a dummy child path)
            alpha, _ = matcher.find_best_alpha(mother_path, father_path, "dummy_path")
            
            # Get similar children
            children_db_path = "children_embeddings.npy"
            results = matcher.find_similar_children_weighted(
                mother_path=mother_path,
                father_path=father_path,
                alpha=alpha,
                children_db_path=children_db_path,
                top_k=5
            )
            
            # Prepare response
            response = {
                'success': True,
                'mother_image': mother_filename,
                'father_image': father_filename,
                'alpha': round(alpha, 2),
                'results': results
            }
            
            return jsonify(response)
            
        except Exception as e:
            return jsonify({'error': f'Error processing images: {str(e)}'}), 500
    
    except Exception as e:
        return jsonify({'error': f'Error saving files: {str(e)}'}), 500

# Serve React App
@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

# Serve uploaded files
@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve children images
@app.route('/children/<path:filename>')
def children_file(filename):
    return send_from_directory('children_db', filename)

if __name__ == '__main__':
    app.run(debug=True)
