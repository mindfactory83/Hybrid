# Voice Biometric Authentication System

## Overview

This is a Flask-based voice biometric authentication system that uses MFCC (Mel-Frequency Cepstral Coefficients) feature extraction and machine learning for user voice recognition. The system provides multi-factor authentication by combining traditional password-based login with voice verification.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Backend Architecture
- **Framework**: Flask with SQLAlchemy ORM
- **Database**: SQLite (configurable via DATABASE_URL environment variable)
- **Authentication**: Flask-Login for session management
- **Audio Processing**: librosa for MFCC feature extraction
- **Machine Learning**: scikit-learn for voice matching algorithms

### Frontend Architecture
- **Template Engine**: Jinja2 with Flask
- **CSS Framework**: Bootstrap 5 (dark theme)
- **JavaScript**: Vanilla JS for audio recording and real-time visualization
- **Icons**: Font Awesome
- **Audio API**: Web Audio API with MediaRecorder

### Security Features
- Password hashing using Werkzeug
- Session-based authentication with Flask-Login
- File upload validation and size limits (16MB)
- Proxy fix for deployment environments

## Key Components

### Audio Processing Pipeline
1. **AudioProcessor** (`audio_processor.py`):
   - Pre-emphasis filtering
   - Silence removal
   - MFCC feature extraction (13 coefficients)
   - Feature normalization
   - Sample rate standardization (16kHz)

2. **VoiceMatcher** (`voice_matcher.py`):
   - Voice sample storage and management
   - Voiceprint creation from multiple samples
   - Cosine similarity matching
   - Configurable similarity threshold (0.75)

### Database Models
1. **User**: Core user information with password hashing
2. **Voiceprint**: Voice sample storage with feature paths
3. **AuthAttempt**: Authentication logging with confidence scores

### Web Interface
- User registration and login
- Voice enrollment (3 samples required)
- Real-time voice authentication
- Dashboard with enrollment status
- Audio visualization during recording

## Data Flow

### Voice Enrollment Process
1. User records 3 voice samples through web interface
2. Audio is processed to extract MFCC features
3. Features are stored as pickle files in `voiceprints/` directory
4. Consolidated voiceprint is created for matching

### Authentication Process
1. User provides username/password (traditional auth)
2. System prompts for voice authentication
3. Audio is recorded and processed in real-time
4. Features are compared against stored voiceprint
5. Authentication succeeds if similarity > threshold

### File Storage Structure
```
voiceprints/
├── user_1_samples/
│   ├── sample_1.pkl
│   ├── sample_2.pkl
│   └── sample_3.pkl
└── user_1_voiceprint.pkl
```

## External Dependencies

### Python Libraries
- **Flask**: Web framework and routing
- **SQLAlchemy**: Database ORM
- **librosa**: Audio processing and MFCC extraction
- **scikit-learn**: Machine learning algorithms
- **numpy/scipy**: Numerical computing
- **werkzeug**: Security utilities

### Frontend Libraries
- **Bootstrap 5**: UI framework
- **Font Awesome**: Icon library
- **Web Audio API**: Browser audio recording

### Audio Requirements
- Supported formats: WAV, MP3, FLAC, M4A, OGG
- Sample rate: 16kHz (standardized)
- Recording duration: 2-5 seconds recommended

## Deployment Strategy

### Environment Configuration
- `SESSION_SECRET`: Flask session security key
- `DATABASE_URL`: Database connection string (defaults to SQLite)
- `UPLOAD_FOLDER`: File upload directory

### File System Requirements
- `uploads/`: Temporary audio file storage
- `voiceprints/`: Voice sample and model storage
- Write permissions for audio processing

### Production Considerations
- ProxyFix middleware for reverse proxy deployment
- Database connection pooling configured
- File size limits enforced (16MB)
- Logging configured for debugging

### Scalability Notes
- SQLite suitable for development/small deployments
- Database can be switched to PostgreSQL/MySQL via DATABASE_URL
- Voice samples stored as files (could be moved to cloud storage)
- Current architecture supports single-server deployment