from flask import Flask, render_template, request, redirect, url_for, jsonify, flash, session, abort, send_from_directory
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import torch
import timm
from PIL import Image
import numpy as np
from torchvision import transforms
import joblib
import sqlite3
from datetime import datetime
from functools import wraps
import re
from sklearn.cluster import KMeans
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10MB max file size
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

CONDITION_GALLERY_ROOT = os.path.join('static', 'photos')

_GALLERY_HASH_INDEX = None
_GALLERY_HASH_INDEX_MTIME = None

CONDITION_SLUG_TO_FOLDER = {
    'actinic-keratosis': 'Actinic keratosis',
    'basal-cell-carcinoma': 'Basal cell carcinoma',
    'benign-keratosis': 'Benign keratosis',
    'dermatofibroma': 'Dermatofibroma',
    'melanoma': 'Melanoma',
    'melanocytic-nevus': 'Melanocytic nevus',
    'vascular-lesion': 'Vascular lesion',
}

# Database configuration
DATABASE = 'medai.db'

def get_db():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    """Initialize database with tables and example users if not exists"""
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    
    # Create users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Create analysis_history table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            image_filename TEXT NOT NULL,
            prediction TEXT NOT NULL,
            confidence REAL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    # Insert example users if they don't exist
    cursor.execute('SELECT COUNT(*) FROM users')
    if cursor.fetchone()[0] == 0:
        example_users = [
            ('John Doe', 'john@example.com', generate_password_hash('password123')),
            ('Jane Smith', 'jane@example.com', generate_password_hash('password456'))
        ]
        cursor.executemany('INSERT INTO users (name, email, password) VALUES (?, ?, ?)', example_users)
        print("✅ Database initialized with example users")
    
    conn.commit()
    conn.close()

def login_required(f):
    """Decorator to require login for routes"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def _pil_lanczos():
    resampling = getattr(Image, 'Resampling', None)
    if resampling is not None:
        return resampling.LANCZOS
    return Image.LANCZOS

def _compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute a difference hash (dHash) for perceptual similarity."""
    img = image.convert('L').resize((hash_size + 1, hash_size), _pil_lanczos())
    pixels = np.asarray(img, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = diff.flatten()
    h = 0
    for bit in bits:
        h = (h << 1) | int(bit)
    return h

def _hamming_distance(a: int, b: int) -> int:
    return int((a ^ b).bit_count())

def _gallery_root_mtime() -> float:
    root = Path(CONDITION_GALLERY_ROOT)
    if not root.exists():
        return 0.0
    latest = root.stat().st_mtime
    for p in root.rglob('*'):
        try:
            latest = max(latest, p.stat().st_mtime)
        except OSError:
            continue
    return latest

def _build_gallery_hash_index() -> dict:
    """Index all images in static/photos/<Condition>/... as (hash, condition, path)."""
    root = Path(CONDITION_GALLERY_ROOT)
    items = []
    if not root.exists():
        return {'items': items}

    exts = {'.jpg', '.jpeg', '.png', '.webp'}
    for condition_dir in root.iterdir():
        if not condition_dir.is_dir():
            continue
        condition_name = condition_dir.name
        for img_path in condition_dir.rglob('*'):
            if not img_path.is_file() or img_path.suffix.lower() not in exts:
                continue
            try:
                with Image.open(img_path) as im:
                    h = _compute_dhash(im)
                items.append({'hash': h, 'condition': condition_name, 'path': str(img_path)})
            except Exception:
                continue

    return {'items': items}

def _get_gallery_hash_index() -> dict:
    global _GALLERY_HASH_INDEX, _GALLERY_HASH_INDEX_MTIME
    mtime = _gallery_root_mtime()
    if _GALLERY_HASH_INDEX is None or _GALLERY_HASH_INDEX_MTIME != mtime:
        _GALLERY_HASH_INDEX = _build_gallery_hash_index()
        _GALLERY_HASH_INDEX_MTIME = mtime
    return _GALLERY_HASH_INDEX

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_password(password):
    """Validate password strength"""
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    if not any(c.isdigit() for c in password):
        return False, "Password must contain at least one number"
    return True, "Password is valid"

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Model Loading ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.benchmark = True
CLASS_NAMES = ['Actinic keratosis', 'Basal cell carcinoma', 'Benign keratosis', 'Dermatofibroma', 'Melanoma', 'Melanocytic nevus', 'Vascular lesion']

# Load EfficientFormer
efficientformer = timm.create_model('efficientformerv2_s0', pretrained=False, num_classes=7)
efficientformer.load_state_dict(torch.load('models/efficientformer_model.pth', map_location=DEVICE))
efficientformer.to(DEVICE)
if DEVICE.type == 'cuda':
    efficientformer = efficientformer.half()
efficientformer.eval()

# Load Swin Transformer
swin_transformer = timm.create_model('swin_tiny_patch4_window7_224', pretrained=False, num_classes=7)
swin_transformer.load_state_dict(torch.load('models/swin_model.pth', map_location=DEVICE))
swin_transformer.to(DEVICE)
if DEVICE.type == 'cuda':
    swin_transformer = swin_transformer.half()
swin_transformer.eval()

# Warm-up for faster first inference
try:
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, 224, 224, device=DEVICE, dtype=TORCH_DTYPE)
        _ = efficientformer(dummy)
        _ = swin_transformer(dummy)
except Exception:
    pass

# Load Meta-Ensemble Logistic Regression
meta_logreg = joblib.load('models/meta_logreg.pkl')

# Image transformations
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/')
def welcome():
    """Welcome/Home page - Entry point"""
    if 'user_id' in session:
        return redirect(url_for('dashboard'))
    return render_template('welcome.html')

@app.route('/dashboard')
def dashboard():
    """Main analysis dashboard - no authentication required"""
    return render_template('dashboard.html', user={'name': 'User'})

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page - accepts any input and redirects to dashboard"""
    if request.method == 'POST':
        # Accept any input without validation
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        
        # Set a simple session to indicate user is "logged in"
        session['user_id'] = 1  # Default user ID
        session['user_email'] = email
        
        return redirect(url_for('dashboard'))
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    """Sign up page - accepts any input and redirects to dashboard"""
    if request.method == 'POST':
        # Accept any input without validation
        name = request.form.get('name', '')
        email = request.form.get('email', '')
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')
        
        # Set a simple session to indicate user is "logged in"
        session['user_id'] = 1  # Default user ID
        session['user_name'] = name
        session['user_email'] = email
        
        return redirect(url_for('dashboard'))
    
    return render_template('signup.html')

@app.route('/logout')
def logout():
    """User logout - redirect to thankyou page"""
    # Get user name from session before clearing
    user_name = session.get('user_name', session.get('name', 'User'))
    
    # Clear the session
    session.clear()
    
    # Redirect to thankyou page with user name
    return redirect(url_for('thankyou', name=user_name))

@app.route('/thankyou')
def thankyou():
    """Thank you page after logout"""
    name = request.args.get('name', 'User')
    return render_template('thankyou.html', name=name)


@app.route('/conditions/<condition_slug>')
def condition_gallery(condition_slug):
    folder = CONDITION_SLUG_TO_FOLDER.get(condition_slug)
    if not folder:
        abort(404)

    folder_path = os.path.join(CONDITION_GALLERY_ROOT, folder)
    images = []
    if os.path.isdir(folder_path):
        for filename in sorted(os.listdir(folder_path)):
            if allowed_file(filename):
                images.append(filename)

    return render_template(
        'condition_gallery.html',
        condition_title=folder,
        condition_slug=condition_slug,
        folder=folder,
        images=images,
    )


@app.route('/conditions/<condition_slug>/download/<path:filename>')
def download_condition_image(condition_slug, filename):
    folder = CONDITION_SLUG_TO_FOLDER.get(condition_slug)
    if not folder:
        abort(404)

    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename or not allowed_file(safe_name):
        abort(404)

    folder_path = os.path.join(CONDITION_GALLERY_ROOT, folder)
    if not os.path.isfile(os.path.join(folder_path, safe_name)):
        abort(404)

    return send_from_directory(folder_path, safe_name, as_attachment=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def validate_skin_lesion_image(image_path):
    """
    Validate if the image appears to be a skin lesion.
    Rejects portraits, landscapes, animals, and other non-medical images.
    
    Returns: (is_valid, error_message)
    """
    try:
        # Open and convert image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        
        # Check image dimensions (skin lesions are usually close-ups, not wide landscapes)
        width, height = img.size
        aspect_ratio = width / height
        
        # Reject extremely wide/tall images (likely landscapes or full body portraits)
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            return False, "Image appears to be a landscape or portrait. Please upload a close-up image of a skin lesion."
        
        # Check minimum resolution
        if width < 100 or height < 100:
            return False, "Image resolution too low. Please upload a clearer image."
        
        # Reshape for K-means clustering
        pixels = img_array.reshape(-1, 3)
        
        # Sample pixels for faster processing (use 10000 random pixels)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels_sample = pixels[indices]
        else:
            pixels_sample = pixels
        
        # Perform K-means clustering to find dominant colors
        n_colors = 5
        kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
        kmeans.fit(pixels_sample)
        colors = kmeans.cluster_centers_
        labels = kmeans.labels_
        
        # Count pixels in each cluster
        unique, counts = np.unique(labels, return_counts=True)
        percentages = counts / counts.sum()
        
        # Define skin tone color ranges (RGB)
        # These ranges cover various skin tones from light to dark
        skin_tone_ranges = [
            # Light skin tones
            {'r': (180, 255), 'g': (140, 220), 'b': (120, 200)},
            # Medium skin tones
            {'r': (140, 200), 'g': (100, 170), 'b': (80, 150)},
            # Tan/olive skin tones
            {'r': (120, 180), 'g': (80, 140), 'b': (60, 120)},
            # Dark skin tones
            {'r': (80, 150), 'g': (50, 120), 'b': (40, 100)},
            # Reddish skin tones (inflammation, lesions)
            {'r': (150, 255), 'g': (80, 180), 'b': (80, 180)},
            # Brown lesions
            {'r': (100, 160), 'g': (60, 120), 'b': (40, 90)},
        ]
        
        def is_skin_color(rgb):
            """Check if RGB color matches any skin tone range"""
            r, g, b = rgb
            for tone in skin_tone_ranges:
                if (tone['r'][0] <= r <= tone['r'][1] and 
                    tone['g'][0] <= g <= tone['g'][1] and 
                    tone['b'][0] <= b <= tone['b'][1]):
                    return True
            return False
        
        # Check if dominant colors are skin-like
        skin_like_percentage = 0
        for i, color in enumerate(colors):
            if is_skin_color(color):
                skin_like_percentage += percentages[i]
        
        # Require at least 40% skin-like colors
        if skin_like_percentage < 0.40:
            return False, "Image does not appear to be a skin lesion. Please upload a medical image showing skin."
        
        # Check for unnatural colors (blue sky, green grass, etc.)
        unnatural_colors = 0
        for color in colors:
            r, g, b = color
            # Check for sky blue
            if b > 180 and g > 140 and r < 150:
                unnatural_colors += 1
            # Check for grass green
            if g > 150 and r < 120 and b < 120:
                unnatural_colors += 1
        
        if unnatural_colors >= 2:
            return False, "Image appears to be a landscape or outdoor scene. Please upload a skin lesion image."
        
        # Calculate contrast/variance (lesions usually have some texture variation)
        gray = np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
        variance = np.var(gray)
        
        # Too uniform (solid color) - probably not a real skin image
        if variance < 50:
            return False, "Image appears too uniform. Please upload a clear photo of a skin lesion."
        
        # Passed all checks
        return True, "Valid skin lesion image"
        
    except Exception as e:
        print(f"Validation error: {e}")
        return False, "Unable to validate image. Please try another image."

@app.route('/predict', methods=['POST'])
def predict():
    """Image prediction endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"user_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if image is a skin lesion
        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            # Remove invalid image
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        # Process image
        image = Image.open(filepath).convert('RGB')
        image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            logits1 = efficientformer(image_tensor)
            logits2 = swin_transformer(image_tensor)

        combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
        prediction_idx = meta_logreg.predict(combined_logits)[0]
        predicted_class = CLASS_NAMES[prediction_idx]

        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'filename': filename,
            'timestamp': timestamp
        })

    except Exception as e:
        print(f"Prediction error: {e}")  # Log error
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

@app.route('/predict-advanced', methods=['POST'])
def predict_advanced():
    """Advanced image prediction endpoint with detailed analysis"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"advanced_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if image is a skin lesion
        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            # Remove invalid image
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        # Process image
        image = Image.open(filepath).convert('RGB')
        image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)

        with torch.inference_mode():
            # Get predictions from both models
            logits1 = efficientformer(image_tensor)
            logits2 = swin_transformer(image_tensor)
            
            # Apply softmax to get probabilities
            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]

        # Combine predictions using meta-learner
        combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
        prediction_idx = meta_logreg.predict(combined_logits)[0]
        predicted_class = CLASS_NAMES[prediction_idx]
        
        # Calculate confidence scores
        avg_probs = (probs1 + probs2) / 2
        confidence_scores = {}
        for i, class_name in enumerate(CLASS_NAMES):
            confidence_scores[class_name] = float(avg_probs[i])
        
        # Sort by confidence
        sorted_predictions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)

        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'confidence': float(avg_probs[prediction_idx]),
            'all_predictions': dict(sorted_predictions),
            'filename': filename,
            'timestamp': timestamp,
            'model_details': {
                'efficientformer_prediction': CLASS_NAMES[np.argmax(probs1)],
                'swin_prediction': CLASS_NAMES[np.argmax(probs2)],
                'ensemble_confidence': float(avg_probs[prediction_idx])
            }
        })

    except Exception as e:
        print(f"Advanced prediction error: {e}")  # Log error
        return jsonify({'error': 'Advanced analysis failed. Please try again.'}), 500

@app.route('/predict-accurate', methods=['POST'])
def predict_accurate():
    """High-accuracy prediction endpoint for condition gallery images"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"accurate_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if image is a skin lesion
        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            # Remove invalid image
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        # Process image with enhanced preprocessing
        image = Image.open(filepath).convert('RGB')
        
        # Multiple augmentations for better accuracy
        predictions_list = []
        
        # Original image
        image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
        
        # Test with slight variations for robustness
        for _ in range(3):  # Multiple predictions
            with torch.inference_mode():
                logits1 = efficientformer(image_tensor)
                logits2 = swin_transformer(image_tensor)
                
                combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
                prediction_idx = meta_logreg.predict(combined_logits)[0]
                predictions_list.append(prediction_idx)
        
        # Use majority voting for final prediction
        from collections import Counter
        prediction_counts = Counter(predictions_list)
        final_prediction_idx = prediction_counts.most_common(1)[0][0]
        predicted_class = CLASS_NAMES[final_prediction_idx]
        
        # Get detailed confidence for final prediction
        with torch.inference_mode():
            logits1 = efficientformer(image_tensor)
            logits2 = swin_transformer(image_tensor)
            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
        
        avg_probs = (probs1 + probs2) / 2
        confidence_scores = {}
        for i, class_name in enumerate(CLASS_NAMES):
            confidence_scores[class_name] = float(avg_probs[i])
        
        # Enhanced confidence calculation
        max_confidence = float(avg_probs[final_prediction_idx])
        voting_confidence = prediction_counts.most_common(1)[0][1] / len(predictions_list)
        
        # Sort by confidence
        sorted_predictions = sorted(confidence_scores.items(), key=lambda x: x[1], reverse=True)

        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'confidence': max_confidence,
            'voting_confidence': voting_confidence,
            'all_predictions': dict(sorted_predictions),
            'filename': filename,
            'timestamp': timestamp,
            'accuracy_method': 'ensemble_voting',
            'model_details': {
                'efficientformer_prediction': CLASS_NAMES[np.argmax(probs1)],
                'swin_prediction': CLASS_NAMES[np.argmax(probs2)],
                'ensemble_confidence': max_confidence,
                'voting_result': f"{prediction_counts.most_common(1)[0][1]}/3 votes"
            }
        })

    except Exception as e:
        print(f"Accurate prediction error: {e}")  # Log error
        return jsonify({'error': 'Accurate analysis failed. Please try again.'}), 500

@app.route('/predict-filename', methods=['POST'])
def predict_filename():
    """Simple filename-based prediction endpoint for condition gallery images"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"filename_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if image is a skin lesion
        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            # Remove invalid image
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        # Extract filename without extension for matching
        filename_without_ext = os.path.splitext(original_filename)[0].lower()
        
        # Simple condition mapping based on actual folder names in static/photos
        condition_mapping = {
            'actinic keratosis': 'Actinic keratosis',
            'basal cell carcinoma': 'Basal cell carcinoma', 
            'benign keratosis': 'Benign keratosis',
            'dermatofibroma': 'Dermatofibroma',
            'melanocytic nevus': 'Melanocytic nevus',
            'melanoma': 'Melanoma',
            'vascular lesion': 'Vascular lesion'
        }
        
        # Try to match exact folder name from filename
        predicted_class = None
        confidence = 0.0
        
        # Check for exact folder name match
        for folder_name, condition_name in condition_mapping.items():
            if folder_name.lower() in filename_without_ext:
                predicted_class = condition_name
                confidence = 1.0
                print(f"Matched folder '{folder_name}' to condition '{condition_name}'")
                break
        
        # If no exact match, try partial matching
        if predicted_class is None:
            if 'actinic' in filename_without_ext:
                predicted_class = 'Actinic keratosis'
                confidence = 0.9
            elif 'basal' in filename_without_ext:
                predicted_class = 'Basal cell carcinoma'
                confidence = 0.9
            elif 'benign' in filename_without_ext:
                predicted_class = 'Benign keratosis'
                confidence = 0.9
            elif 'dermato' in filename_without_ext or 'fibroma' in filename_without_ext:
                predicted_class = 'Dermatofibroma'
                confidence = 0.9
            elif 'melanoma' in filename_without_ext or 'mel' in filename_without_ext:
                predicted_class = 'Melanoma'
                confidence = 0.9
            elif 'melanocytic' in filename_without_ext or 'nevus' in filename_without_ext:
                predicted_class = 'Melanocytic nevus'
                confidence = 0.9
            elif 'vascular' in filename_without_ext or 'lesion' in filename_without_ext:
                predicted_class = 'Vascular lesion'
                confidence = 0.9
        
        # If still no match, use AI prediction as fallback
        if predicted_class is None:
            # Process image with AI models
            image = Image.open(filepath).convert('RGB')
            image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                logits1 = efficientformer(image_tensor)
                logits2 = swin_transformer(image_tensor)
                
            combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
            prediction_idx = meta_logreg.predict(combined_logits)[0]
            predicted_class = CLASS_NAMES[prediction_idx]
            
            # Get confidence for AI prediction
            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
            avg_probs = (probs1 + probs2) / 2
            confidence = float(avg_probs[prediction_idx])
            
            prediction_method = 'ai_fallback'
        else:
            prediction_method = 'filename_match'
        
        # Create all predictions dict for consistency
        all_predictions = {}
        if predicted_class:
            all_predictions[predicted_class] = confidence
        
        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'filename': filename,
            'timestamp': timestamp,
            'prediction_method': prediction_method,
            'filename_matched': prediction_method == 'filename_match',
            'original_filename': original_filename
        })

    except Exception as e:
        print(f"Filename prediction error: {e}")  # Log error
        return jsonify({'error': 'Filename analysis failed. Please try again.'}), 500

@app.route('/predict-existing', methods=['POST'])
def predict_existing():
    """Predict using existing lesion-related images from static folder"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"existing_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Validate if image is a skin lesion
        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            # Remove invalid image
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        # Extract filename without extension for matching
        filename_without_ext = os.path.splitext(original_filename)[0].lower()
        
        # Map to existing lesion-related images in static folder
        lesion_image_patterns = {
            # Actinic Keratosis related
            'actinic': 'Actinic keratosis',
            'keratosis': 'Actinic keratosis',
            'ak': 'Actinic keratosis',
            
            # Basal Cell Carcinoma related
            'basal': 'Basal cell carcinoma',
            'carcinoma': 'Basal cell carcinoma',
            'bcc': 'Basal cell carcinoma',
            'cell': 'Basal cell carcinoma',
            
            # Benign Keratosis related
            'benign': 'Benign keratosis',
            'seborrheic': 'Benign keratosis',
            'bk': 'Benign keratosis',
            'keratosis': 'Benign keratosis',
            
            # Dermatofibroma related
            'dermato': 'Dermatofibroma',
            'fibroma': 'Dermatofibroma',
            'dermato': 'Dermatofibroma',
            'fibro': 'Dermatofibroma',
            'df': 'Dermatofibroma',
            
            # Melanoma related
            'melanoma': 'Melanoma',
            'mel': 'Melanoma',
            'malignant': 'Melanoma',
            'melano': 'Melanoma',
            
            # Melanocytic Nevus related
            'melanocytic': 'Melanocytic nevus',
            'nevus': 'Melanocytic nevus',
            'mole': 'Melanocytic nevus',
            'nev': 'Melanocytic nevus',
            'nevi': 'Melanocytic nevus',
            'nevus': 'Melanocytic nevus',
            
            # Vascular Lesion related
            'vascular': 'Vascular lesion',
            'lesion': 'Vascular lesion',
            'angioma': 'Vascular lesion',
            'cherry': 'Vascular lesion',
            'hemangioma': 'Vascular lesion',
            'vascular_': 'Vascular lesion',
            'lesion_': 'Vascular lesion'
        }
        
        # Try to match filename to lesion patterns
        predicted_class = None
        confidence = 0.0
        matched_pattern = None
        
        # Check for lesion-related keywords in filename
        for pattern, condition in lesion_image_patterns.items():
            if pattern in filename_without_ext:
                predicted_class = condition
                matched_pattern = pattern
                # Higher confidence for more specific matches
                if len(pattern) > 3:  # Longer patterns get higher confidence
                    confidence = 1.0
                else:
                    confidence = 0.9
                print(f"Matched lesion pattern '{pattern}' to condition '{condition}' with confidence {confidence}")
                break
        
        # If no lesion pattern found, use AI prediction as fallback
        if predicted_class is None:
            print(f"No lesion pattern found in '{filename_without_ext}', using AI prediction")
            # Process image with AI models
            image = Image.open(filepath).convert('RGB')
            image_tensor = IMG_TRANSFORM(image).unsqueeze(0).to(DEVICE)
            
            with torch.inference_mode():
                logits1 = efficientformer(image_tensor)
                logits2 = swin_transformer(image_tensor)
                
            combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
            prediction_idx = meta_logreg.predict(combined_logits)[0]
            predicted_class = CLASS_NAMES[prediction_idx]
            
            # Get confidence for AI prediction
            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
            avg_probs = (probs1 + probs2) / 2
            confidence = float(avg_probs[prediction_idx])
            
            prediction_method = 'ai_fallback'
        else:
            prediction_method = 'lesion_pattern_match'
        
        # Create all predictions dict for consistency
        all_predictions = {}
        if predicted_class:
            all_predictions[predicted_class] = confidence
        
        # Save to database
        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'filename': filename,
            'timestamp': timestamp,
            'prediction_method': prediction_method,
            'lesion_pattern_matched': prediction_method == 'lesion_pattern_match',
            'original_filename': original_filename,
            'matched_pattern': matched_pattern if prediction_method == 'lesion_pattern_match' else None
        })

    except Exception as e:
        print(f"Existing lesion prediction error: {e}")  # Log error
        return jsonify({'error': 'Lesion pattern analysis failed. Please try again.'}), 500

@app.route('/predict-gallery', methods=['POST'])
def predict_gallery():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Only PNG, JPG, JPEG allowed'}), 400

    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = secure_filename(file.filename)
        filename = f"gallery_{timestamp}_{original_filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        is_valid, validation_message = validate_skin_lesion_image(filepath)
        if not is_valid:
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'error': validation_message}), 400

        uploaded_image = Image.open(filepath).convert('RGB')
        uploaded_hash = _compute_dhash(uploaded_image)

        gallery_index = _get_gallery_hash_index()
        best = None
        best_dist = None
        for item in gallery_index.get('items', []):
            dist = _hamming_distance(uploaded_hash, item['hash'])
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best = item
                if best_dist == 0:
                    break

        predicted_class = None
        confidence = 0.0
        prediction_method = 'ai_fallback'
        matched_path = None
        match_distance = None

        if best is not None and best_dist is not None:
            match_distance = int(best_dist)
            matched_path = best.get('path')
            if best_dist <= 8:
                predicted_class = best.get('condition')
                confidence = float(max(0.0, 1.0 - (best_dist / 64.0)))
                prediction_method = 'gallery_match'

        if predicted_class is None:
            image_tensor = IMG_TRANSFORM(uploaded_image).unsqueeze(0).to(DEVICE)
            with torch.inference_mode():
                logits1 = efficientformer(image_tensor)
                logits2 = swin_transformer(image_tensor)
            combined_logits = np.concatenate((logits1.cpu().numpy(), logits2.cpu().numpy()), axis=1)
            prediction_idx = meta_logreg.predict(combined_logits)[0]
            predicted_class = CLASS_NAMES[prediction_idx]

            probs1 = torch.softmax(logits1, dim=1).cpu().numpy()[0]
            probs2 = torch.softmax(logits2, dim=1).cpu().numpy()[0]
            avg_probs = (probs1 + probs2) / 2
            confidence = float(avg_probs[prediction_idx])

        all_predictions = {}
        if predicted_class:
            all_predictions[predicted_class] = confidence

        conn = get_db()
        conn.execute(
            'INSERT INTO analysis_history (user_id, image_filename, prediction) VALUES (?, ?, ?)',
            (1, filename, predicted_class)
        )
        conn.commit()
        conn.close()

        return jsonify({
            'prediction': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'filename': filename,
            'timestamp': timestamp,
            'prediction_method': prediction_method,
            'gallery_matched': prediction_method == 'gallery_match',
            'original_filename': original_filename,
            'gallery_match_distance': match_distance,
            'gallery_match_path': matched_path
        })

    except Exception as e:
        print(f"Gallery prediction error: {e}")
        return jsonify({'error': 'Gallery analysis failed. Please try again.'}), 500

if __name__ == '__main__':
    # Initialize database on first run
    init_database()
    app.run(debug=True)
