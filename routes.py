import os
import logging
from flask import render_template, request, redirect, url_for, flash, session, jsonify
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.utils import secure_filename
from app import app, db
from models import User, Voiceprint, AuthAttempt
from audio_processor import AudioProcessor
from voice_matcher import VoiceMatcher
import tempfile

logger = logging.getLogger(__name__)

# Initialize processors
audio_processor = AudioProcessor()
voice_matcher = VoiceMatcher()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'wav', 'mp3', 'flac', 'm4a', 'ogg'}

@app.route('/')
def index():
    if current_user.is_authenticated:
        return redirect(url_for('dashboard'))
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        # Check if user already exists
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return render_template('register.html')
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return render_template('register.html')
        
        # Create new user
        user = User(username=username, email=email)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please enroll your voice for biometric authentication.', 'success')
        login_user(user)
        return redirect(url_for('enroll_voice'))
    
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        
        if user and user.check_password(password):
            login_user(user)
            if user.has_voiceprint():
                flash('Password authentication successful. Please authenticate with your voice.', 'info')
                return redirect(url_for('voice_auth'))
            else:
                flash('Please enroll your voice for biometric authentication.', 'warning')
                return redirect(url_for('enroll_voice'))
        else:
            flash('Invalid username or password', 'error')
    
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out', 'info')
    return redirect(url_for('index'))

@app.route('/dashboard')
@login_required
def dashboard():
    # Get recent auth attempts
    recent_attempts = AuthAttempt.query.filter_by(user_id=current_user.id).order_by(AuthAttempt.timestamp.desc()).limit(10).all()
    return render_template('dashboard.html', recent_attempts=recent_attempts)

@app.route('/enroll_voice')
@login_required
def enroll_voice():
    return render_template('enroll_voice.html')

@app.route('/upload_voice_sample', methods=['POST'])
@login_required
def upload_voice_sample():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        file = request.files['audio']
        sample_number = int(request.form.get('sample_number', 1))
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process the audio and extract MFCC features
            features = audio_processor.extract_mfcc_features(temp_path)
            
            if features is None:
                return jsonify({'success': False, 'error': 'Failed to extract features from audio'})
            
            # Save features for this sample
            voice_matcher.save_voice_sample(current_user.id, sample_number, features)
            
            # Check if we have enough samples to create the voiceprint
            total_samples = voice_matcher.get_sample_count(current_user.id)
            
            if total_samples >= 3:  # Require at least 3 samples
                # Create the final voiceprint
                success = voice_matcher.create_voiceprint(current_user.id)
                if success:
                    current_user.is_voice_enrolled = True
                    db.session.commit()
                    return jsonify({
                        'success': True, 
                        'message': 'Voice enrollment complete!',
                        'enrolled': True,
                        'sample_count': total_samples
                    })
            
            return jsonify({
                'success': True, 
                'message': f'Sample {sample_number} recorded successfully',
                'enrolled': False,
                'sample_count': total_samples,
                'samples_needed': max(0, 3 - total_samples)
            })
            
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error processing voice sample: {str(e)}")
        return jsonify({'success': False, 'error': 'Failed to process audio sample'})

@app.route('/voice_auth')
@login_required
def voice_auth():
    if not current_user.has_voiceprint():
        flash('Please enroll your voice first', 'warning')
        return redirect(url_for('enroll_voice'))
    return render_template('voice_auth.html')

@app.route('/authenticate_voice', methods=['POST'])
@login_required
def authenticate_voice():
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'No audio file provided'})
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        # Save the uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            file.save(temp_file.name)
            temp_path = temp_file.name
        
        try:
            # Process the audio and extract MFCC features
            features = audio_processor.extract_mfcc_features(temp_path)
            
            if features is None:
                return jsonify({'success': False, 'error': 'Failed to extract features from audio'})
            
            # Compare with stored voiceprint
            is_match, confidence = voice_matcher.authenticate_voice(current_user.id, features)
            
            # Log the attempt
            attempt = AuthAttempt(
                user_id=current_user.id,
                success=is_match,
                confidence_score=confidence,
                ip_address=request.remote_addr
            )
            db.session.add(attempt)
            db.session.commit()
            
            if is_match:
                session['voice_authenticated'] = True
                return jsonify({
                    'success': True,
                    'authenticated': True,
                    'confidence': confidence,
                    'message': 'Voice authentication successful!'
                })
            else:
                return jsonify({
                    'success': True,
                    'authenticated': False,
                    'confidence': confidence,
                    'message': 'Voice authentication failed. Please try again.'
                })
                
        finally:
            # Clean up temporary file
            if os.path.exists(temp_path):
                os.unlink(temp_path)
                
    except Exception as e:
        logger.error(f"Error during voice authentication: {str(e)}")
        return jsonify({'success': False, 'error': 'Authentication failed due to processing error'})

@app.route('/re_enroll')
@login_required
def re_enroll():
    """Allow user to re-enroll their voice"""
    # Clear existing voiceprint
    voice_matcher.clear_user_voiceprint(current_user.id)
    current_user.is_voice_enrolled = False
    db.session.commit()
    flash('Previous voice enrollment cleared. Please enroll again.', 'info')
    return redirect(url_for('enroll_voice'))
