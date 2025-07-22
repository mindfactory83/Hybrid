from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import os

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_voice_enrolled = db.Column(db.Boolean, default=False)
    
    # Relationship to voiceprints
    voiceprints = db.relationship('Voiceprint', backref='user', lazy='dynamic', cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_voiceprint_path(self):
        """Get the path to the user's voiceprint file"""
        return f"voiceprints/user_{self.id}_voiceprint.pkl"

    def has_voiceprint(self):
        """Check if user has an enrolled voiceprint"""
        return self.is_voice_enrolled and os.path.exists(self.get_voiceprint_path())

class Voiceprint(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    sample_number = db.Column(db.Integer, nullable=False)  # 1, 2, 3 for multiple samples
    features_path = db.Column(db.String(256), nullable=False)  # Path to MFCC features file
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_primary = db.Column(db.Boolean, default=False)  # Primary voiceprint for matching

class AuthAttempt(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    success = db.Column(db.Boolean, nullable=False)
    confidence_score = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    ip_address = db.Column(db.String(45))
