"""
Firebase client for authentication and storage operations.
Handles user auth, dataset uploads, and model storage.
"""

import os
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, List, Any
import firebase_admin
from firebase_admin import credentials, auth, storage
import streamlit as st


class FirebaseClient:
    def __init__(self):
        """Initialize Firebase client with credentials from environment variables."""
        self.app = None
        self.bucket = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase app and storage bucket."""
        try:
            # Check if Firebase is already initialized
            if firebase_admin._apps:
                self.app = firebase_admin.get_app()
            else:
                # Get credentials from environment variables
                cred_dict = {
                    "type": "service_account",
                    "project_id": os.getenv("FIREBASE_PROJECT_ID"),
                    "private_key_id": os.getenv("FIREBASE_PRIVATE_KEY_ID"),
                    "private_key": os.getenv("FIREBASE_PRIVATE_KEY").replace('\\n', '\n'),
                    "client_email": os.getenv("FIREBASE_CLIENT_EMAIL"),
                    "client_id": os.getenv("FIREBASE_CLIENT_ID"),
                    "auth_uri": os.getenv("FIREBASE_AUTH_URI"),
                    "token_uri": os.getenv("FIREBASE_TOKEN_URI"),
                    "auth_provider_x509_cert_url": os.getenv("FIREBASE_AUTH_PROVIDER_X509_CERT_URL"),
                    "client_x509_cert_url": os.getenv("FIREBASE_CLIENT_X509_CERT_URL")
                }
                
                cred = credentials.Certificate(cred_dict)
                self.app = firebase_admin.initialize_app(cred, {
                    'storageBucket': os.getenv("FIREBASE_STORAGE_BUCKET")
                })
            
            # Initialize storage bucket
            self.bucket = storage.bucket()
            
        except Exception as e:
            st.error(f"Failed to initialize Firebase: {str(e)}")
            st.info("Please check your Firebase credentials in the .env file")
    
    def authenticate_user(self, email: str, password: str) -> Optional[str]:
        """
        Authenticate user with email and password.
        Returns user ID if successful, None otherwise.
        """
        try:
            # For MVP, we'll use a simple approach with Firebase Auth
            # In production, you'd want to implement proper sign-in flow
            user = auth.get_user_by_email(email)
            return user.uid
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None
    
    def create_guest_user(self) -> str:
        """Create a temporary guest user ID."""
        guest_id = f"guest_{uuid.uuid4().hex[:8]}"
        return guest_id
    
    def upload_dataset(self, user_id: str, dataset_name: str, file_data: bytes) -> Optional[str]:
        """
        Upload dataset to Firebase Storage.
        Returns download URL if successful, None otherwise.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}/datasets/{timestamp}_{dataset_name}"
            
            blob = self.bucket.blob(filename)
            blob.upload_from_string(file_data)
            
            # Make the blob publicly accessible for download
            blob.make_public()
            return blob.public_url
            
        except Exception as e:
            st.error(f"Failed to upload dataset: {str(e)}")
            return None
    
    def save_model(self, user_id: str, model_name: str, model_data: bytes, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Save trained model to Firebase Storage.
        Returns download URL if successful, None otherwise.
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{user_id}/models/{timestamp}_{model_name}.pkl"
            
            blob = self.bucket.blob(filename)
            blob.upload_from_string(model_data)
            
            # Save metadata as JSON
            metadata_filename = f"{user_id}/models/{timestamp}_{model_name}_metadata.json"
            metadata_blob = self.bucket.blob(metadata_filename)
            metadata_blob.upload_from_string(json.dumps(metadata))
            
            # Make the blob publicly accessible for download
            blob.make_public()
            return blob.public_url
            
        except Exception as e:
            st.error(f"Failed to save model: {str(e)}")
            return None
    
    def list_user_projects(self, user_id: str) -> List[Dict[str, Any]]:
        """
        List all projects (models) for a user.
        Returns list of project metadata.
        """
        try:
            projects = []
            prefix = f"{user_id}/models/"
            
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if blob.name.endswith('_metadata.json'):
                    # Download metadata
                    metadata_json = blob.download_as_text()
                    metadata = json.loads(metadata_json)
                    
                    # Get corresponding model file
                    model_filename = blob.name.replace('_metadata.json', '.pkl')
                    model_blob = self.bucket.blob(model_filename)
                    
                    if model_blob.exists():
                        projects.append({
                            'name': metadata.get('model_name', 'Unknown'),
                            'accuracy': metadata.get('accuracy', 0),
                            'model_type': metadata.get('model_type', 'Unknown'),
                            'dataset_name': metadata.get('dataset_name', 'Unknown'),
                            'created_at': metadata.get('created_at', ''),
                            'download_url': model_blob.public_url
                        })
            
            # Sort by creation date (newest first)
            projects.sort(key=lambda x: x['created_at'], reverse=True)
            return projects
            
        except Exception as e:
            st.error(f"Failed to list projects: {str(e)}")
            return []
    
    def download_model(self, user_id: str, model_name: str) -> Optional[bytes]:
        """
        Download model file from Firebase Storage.
        Returns model data if successful, None otherwise.
        """
        try:
            # Find the model file
            prefix = f"{user_id}/models/"
            blobs = self.bucket.list_blobs(prefix=prefix)
            
            for blob in blobs:
                if blob.name.endswith('.pkl') and model_name in blob.name:
                    return blob.download_as_bytes()
            
            return None
            
        except Exception as e:
            st.error(f"Failed to download model: {str(e)}")
            return None


# Global Firebase client instance
@st.cache_resource
def get_firebase_client():
    """Get cached Firebase client instance."""
    return FirebaseClient()
