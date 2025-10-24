"""
Local storage client for authentication and storage operations.
Handles user auth, dataset uploads, and model storage using local files.
"""

import os
import json
import uuid
import pickle
from datetime import datetime
from typing import Optional, Dict, List, Any
import streamlit as st


class LocalStorageClient:
    def __init__(self):
        """Initialize local storage client."""
        self.storage_dir = "data"
        self.users_dir = os.path.join(self.storage_dir, "users")
        self._initialize_storage()
    
    def _initialize_storage(self):
        """Create storage directories if they don't exist."""
        try:
            os.makedirs(self.storage_dir, exist_ok=True)
            os.makedirs(self.users_dir, exist_ok=True)
        except Exception as e:
            st.error(f"Failed to initialize storage: {str(e)}")
    
    def authenticate_user(self, email: str, password: str) -> Optional[str]:
        """
        Simple authentication for MVP - just validate email format.
        Returns user ID if successful, None otherwise.
        """
        try:
            # Simple email validation
            if "@" in email and "." in email.split("@")[1]:
                # Create user directory
                user_id = f"user_{hash(email) % 1000000}"
                user_dir = os.path.join(self.users_dir, user_id)
                os.makedirs(user_dir, exist_ok=True)
                
                # Save user info
                user_info = {
                    "email": email,
                    "created_at": datetime.now().isoformat(),
                    "user_id": user_id
                }
                
                user_file = os.path.join(user_dir, "user_info.json")
                with open(user_file, 'w') as f:
                    json.dump(user_info, f)
                
                return user_id
            else:
                st.error("Please enter a valid email address")
                return None
        except Exception as e:
            st.error(f"Authentication failed: {str(e)}")
            return None
    
    def create_guest_user(self) -> str:
        """Create a temporary guest user ID."""
        guest_id = f"guest_{uuid.uuid4().hex[:8]}"
        guest_dir = os.path.join(self.users_dir, guest_id)
        os.makedirs(guest_dir, exist_ok=True)
        return guest_id
    
    def upload_dataset(self, user_id: str, dataset_name: str, file_data: bytes) -> Optional[str]:
        """
        Save dataset to local storage.
        Returns file path if successful, None otherwise.
        """
        try:
            user_dir = os.path.join(self.users_dir, user_id)
            datasets_dir = os.path.join(user_dir, "datasets")
            os.makedirs(datasets_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{dataset_name}"
            filepath = os.path.join(datasets_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(file_data)
            
            return filepath
            
        except Exception as e:
            st.error(f"Failed to save dataset: {str(e)}")
            return None
    
    def save_model(self, user_id: str, model_name: str, model_data: bytes, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Save trained model to local storage.
        Returns file path if successful, None otherwise.
        """
        try:
            user_dir = os.path.join(self.users_dir, user_id)
            models_dir = os.path.join(user_dir, "models")
            os.makedirs(models_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{model_name}.pkl"
            filepath = os.path.join(models_dir, filename)
            
            with open(filepath, 'wb') as f:
                f.write(model_data)
            
            # Save metadata as JSON
            metadata_filename = f"{timestamp}_{model_name}_metadata.json"
            metadata_filepath = os.path.join(models_dir, metadata_filename)
            
            with open(metadata_filepath, 'w') as f:
                json.dump(metadata, f)
            
            return filepath
            
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
            user_dir = os.path.join(self.users_dir, user_id)
            models_dir = os.path.join(user_dir, "models")
            
            if not os.path.exists(models_dir):
                return projects
            
            # Find all metadata files
            for filename in os.listdir(models_dir):
                if filename.endswith('_metadata.json'):
                    metadata_filepath = os.path.join(models_dir, filename)
                    
                    try:
                        with open(metadata_filepath, 'r') as f:
                            metadata = json.load(f)
                        
                        # Check if corresponding model file exists
                        model_filename = filename.replace('_metadata.json', '.pkl')
                        model_filepath = os.path.join(models_dir, model_filename)
                        
                        if os.path.exists(model_filepath):
                            projects.append({
                                'name': metadata.get('model_name', 'Unknown'),
                                'accuracy': metadata.get('accuracy', 0),
                                'model_type': metadata.get('model_type', 'Unknown'),
                                'dataset_name': metadata.get('dataset_name', 'Unknown'),
                                'created_at': metadata.get('created_at', ''),
                                'file_path': model_filepath
                            })
                    except Exception as e:
                        continue
            
            # Sort by creation date (newest first)
            projects.sort(key=lambda x: x['created_at'], reverse=True)
            return projects
            
        except Exception as e:
            st.error(f"Failed to list projects: {str(e)}")
            return []
    
    def download_model(self, user_id: str, model_name: str) -> Optional[bytes]:
        """
        Load model file from local storage.
        Returns model data if successful, None otherwise.
        """
        try:
            user_dir = os.path.join(self.users_dir, user_id)
            models_dir = os.path.join(user_dir, "models")
            
            if not os.path.exists(models_dir):
                return None
            
            # Find the model file
            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl') and model_name in filename:
                    filepath = os.path.join(models_dir, filename)
                    with open(filepath, 'rb') as f:
                        return f.read()
            
            return None
            
        except Exception as e:
            st.error(f"Failed to load model: {str(e)}")
            return None


# Global storage client instance
@st.cache_resource
def get_firebase_client():
    """Get cached storage client instance."""
    return LocalStorageClient()
