# Modela - Lean AutoML MVP

🤖 **Upload your dataset and train an AI model in minutes — no coding required.**

Modela is a minimal, fully functional AutoML web tool for non-technical users that allows them to upload tabular datasets, auto-train models, view results, and download trained models. Built with Streamlit and PyCaret, it's designed to be ultra-lean, low-cost, and deployable on free tiers.

## ✨ Features

- **📊 Easy Data Upload**: Support for CSV and Excel files up to 50MB
- **🤖 AutoML Training**: Automatic model training using PyCaret with 5+ algorithms
- **📈 Rich Visualizations**: Feature importance, confusion matrices, and performance metrics
- **💾 Model Download**: Download trained models as .pkl files
- **🔐 User Authentication**: Firebase Auth with guest mode support
- **☁️ Cloud Storage**: Firebase Storage for datasets and models
- **📱 Responsive Design**: Modern, mobile-friendly interface

## 🚀 Quick Start

### Local Development

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Modela
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Firebase**
   - Go to [Firebase Console](https://console.firebase.google.com/)
   - Create a new project
   - Enable Authentication (Email/Password)
   - Enable Storage and create a bucket
   - Go to Project Settings > Service Accounts
   - Generate new private key (downloads JSON file)
   - Copy the values to `.env` file (see `.env.example`)

4. **Create environment file**
   ```bash
   cp .env.example .env
   # Edit .env with your Firebase credentials
   ```

5. **Run the application**
   ```bash
   streamlit run app.py
   ```

6. **Open your browser**
   Navigate to `http://localhost:8501`

### Streamlit Cloud Deployment

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Initial commit"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [Streamlit Cloud](https://share.streamlit.io/)
   - Connect your GitHub account
   - Select your repository
   - Add Firebase secrets in the dashboard:
     - `FIREBASE_PROJECT_ID`
     - `FIREBASE_PRIVATE_KEY_ID`
     - `FIREBASE_PRIVATE_KEY`
     - `FIREBASE_CLIENT_EMAIL`
     - `FIREBASE_CLIENT_ID`
     - `FIREBASE_AUTH_URI`
     - `FIREBASE_TOKEN_URI`
     - `FIREBASE_AUTH_PROVIDER_X509_CERT_URL`
     - `FIREBASE_CLIENT_X509_CERT_URL`
     - `FIREBASE_STORAGE_BUCKET`

3. **Deploy**
   - Click "Deploy"
   - Wait for deployment to complete
   - Your app will be available at `https://your-app-name.streamlit.app`

## 📁 Project Structure

```
Modela/
├── app.py                    # Main Streamlit app entry point
├── pages/
│   ├── 1_upload.py          # Dataset upload page
│   ├── 2_train.py           # Model training page
│   ├── 3_results.py          # Results dashboard
│   └── 4_projects.py         # Past projects list
├── utils/
│   ├── automl.py            # PyCaret wrapper functions
│   ├── firebase_client.py   # Firebase auth + storage
│   ├── preprocessing.py     # Data cleaning/validation
│   └── visualization.py     # Charts and metrics display
├── requirements.txt
├── .streamlit/config.toml   # Streamlit theme config
├── README.md
└── .env.example             # Firebase credentials template
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AutoML**: PyCaret
- **Authentication**: Firebase Auth
- **Storage**: Firebase Storage
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud (free tier)

## 📊 Supported Data Types

- **Classification**: Predicting categories (e.g., spam/not spam, customer churn)
- **Regression**: Predicting numerical values (e.g., house prices, sales forecasts)

## 🔧 Configuration

### Environment Variables

Create a `.env` file with your Firebase credentials:

```env
FIREBASE_PROJECT_ID=your-project-id
FIREBASE_PRIVATE_KEY_ID=your-private-key-id
FIREBASE_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\nyour-private-key\n-----END PRIVATE KEY-----\n"
FIREBASE_CLIENT_EMAIL=your-service-account@your-project.iam.gserviceaccount.com
FIREBASE_CLIENT_ID=your-client-id
FIREBASE_AUTH_URI=https://accounts.google.com/o/oauth2/auth
FIREBASE_TOKEN_URI=https://oauth2.googleapis.com/token
FIREBASE_AUTH_PROVIDER_X509_CERT_URL=https://www.googleapis.com/oauth2/v1/certs
FIREBASE_CLIENT_X509_CERT_URL=https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project.iam.gserviceaccount.com
FIREBASE_STORAGE_BUCKET=your-project.appspot.com
```

### Streamlit Configuration

The `.streamlit/config.toml` file contains:
- Theme colors
- File upload size limit (50MB)
- Font settings

## 🎯 Usage Guide

### 1. Upload Dataset
- Click "Upload Dataset" in the sidebar
- Choose a CSV or Excel file (max 50MB)
- Preview your data and check the summary
- Save the dataset to cloud storage

### 2. Train Model
- Select your target column (what you want to predict)
- Choose problem type (classification or regression)
- Configure advanced settings (optional)
- Click "Start Training" and wait 1-3 minutes

### 3. View Results
- See model performance metrics
- View feature importance charts
- Download your trained model
- Get insights and explanations

### 4. Manage Projects
- View all your trained models
- Download previous models
- Track your progress over time

## 🔒 Security & Privacy

- **Authentication**: Firebase Auth with email/password
- **Guest Mode**: Anonymous users can try the app
- **Data Storage**: All data stored securely in Firebase Storage
- **Privacy**: No data is shared with third parties

## 💰 Cost Structure

- **Free Tier**: 50MB datasets, 1 training at a time
- **Streamlit Cloud**: Free tier (500MB RAM, public repos)
- **Firebase**: Free tier (1GB storage, 10GB transfer)
- **Total Cost**: $0/month for MVP

## 🚀 Future Enhancements

- **Paid Tiers**: Larger datasets, concurrent training, API access
- **Model Deployment**: REST API endpoints for trained models
- **Advanced Features**: Hyperparameter tuning, ensemble methods
- **Integrations**: Export to various formats, cloud platforms

## 🐛 Troubleshooting

### Common Issues

1. **Firebase Connection Error**
   - Check your Firebase credentials in `.env`
   - Ensure Firebase project is properly configured
   - Verify storage bucket exists

2. **Model Training Fails**
   - Check dataset size (max 50MB)
   - Ensure target column has sufficient data
   - Try with a smaller dataset first

3. **Upload Issues**
   - Check file format (CSV/Excel only)
   - Verify file size (max 50MB)
   - Ensure file is not corrupted

### Getting Help

- Check the [Streamlit documentation](https://docs.streamlit.io/)
- Review [PyCaret documentation](https://pycaret.readthedocs.io/)
- Firebase [support resources](https://firebase.google.com/support)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Support

For support, email support@modela.ai or create an issue in the GitHub repository.

---

**Built with ❤️ using Streamlit, PyCaret, and Firebase**
"# modela" 
