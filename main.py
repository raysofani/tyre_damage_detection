import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import tensorflow as tf

from app.model import build_model
from app.analyze import process_image_file, process_video_file

# Configure upload directories
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4'}

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB max file size

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the pre-trained model
MODEL_PATH = 'tire_damage_model.h5'
model = build_model()
model.load_weights(MODEL_PATH)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and prediction."""
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If no file is selected
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # If file is allowed
    if file and allowed_file(file.filename):
        # Secure the filename
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Determine file type and process accordingly
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        try:
            if file_ext in {'png', 'jpg', 'jpeg'}:
                # Image processing
                original_img = cv2.imread(filepath)
                img = cv2.resize(original_img, (128, 128))
                img = img.astype("float32") / 255.0
                img = np.expand_dims(img, axis=0)
                
                # Predict
                prediction = model.predict(img)
                damage_percentage = prediction[0][0]
                
                # Determine damage status
                if damage_percentage < 0.1:
                    damage_status = "Minimal Wear"
                elif damage_percentage < 0.3:
                    damage_status = "Light Wear"
                elif damage_percentage < 0.5:
                    damage_status = "Moderate Wear"
                elif damage_percentage < 0.7:
                    damage_status = "Significant Wear"
                else:
                    damage_status = "Severe Damage"
                
                return jsonify({
                    'damage_percentage': float(damage_percentage * 100),
                    'damage_status': damage_status
                })
            
            elif file_ext == 'mp4':
                # Video processing
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{filename}')
                process_video_file(model, filepath, output_path)
                
                return jsonify({
                    'processed_video': f'uploads/processed_{filename}'
                })
        
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/webcam', methods=['GET'])
def webcam_detection():
    """Render webcam detection page."""
    return render_template('webcam.html')

if __name__ == '__main__':
    app.run(debug=True)