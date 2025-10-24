# Modela - Open Source AutoML Platform

🤖 **Upload your dataset and train an AI model in minutes — no coding required.**

[![GitHub stars](https://img.shields.io/github/stars/youssof20/modela?style=social)](https://github.com/youssof20/modela)
[![GitHub forks](https://img.shields.io/github/forks/youssof20/modela?style=social)](https://github.com/youssof20/modela)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Modela is a powerful, open-source AutoML platform that democratizes machine learning by making it accessible to everyone. Built with Streamlit and PyCaret, it provides an intuitive web interface for training machine learning models without writing a single line of code.

## ✨ Features

- **📊 Easy Data Upload**: Support for CSV and Excel files up to 50MB
- **🤖 AutoML Training**: Automatic model training using PyCaret with 5+ algorithms
- **📈 Rich Visualizations**: Feature importance, confusion matrices, and performance metrics
- **💾 Model Download**: Download trained models as .pkl files
- **🔐 User Management**: Simple local authentication with guest mode support
- **💾 Local Storage**: Local file storage for datasets and models (completely free)
- **📱 Responsive Design**: Modern, mobile-friendly interface
- **🚀 One-Click Deployment**: Deploy to Streamlit Cloud in minutes
- **📚 Open Source**: Fully open source with MIT license
- **🎯 Production Ready**: Battle-tested for real-world use cases

## 🚀 Quick Start

**Modela is designed to run locally on your machine.** This gives you full control over your data and models while keeping everything private and secure.

### Local Installation (Recommended)

1. **Clone the repository**
   ```bash
   git clone https://github.com/youssof20/modela.git
   cd modela
   ```

2. **Check compatibility (recommended)**
   ```bash
   python check_installation.py
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Create demo datasets (optional)**
   ```bash
   python create_demo_data.py
   ```

5. **Test the installation**
   ```bash
   python test_modela.py
   ```

6. **Run the application**
   ```bash
   streamlit run app.py
   ```

7. **Open your browser**
   Navigate to `http://localhost:8501`

### Why Run Locally?

- **🔒 Privacy**: Your data never leaves your machine
- **⚡ Speed**: No network latency, faster training
- **💰 Cost**: Completely free, no cloud costs
- **🛠️ Control**: Full control over your environment
- **📊 Offline**: Works without internet connection

### System Requirements

- **Python**: 3.8, 3.9, or 3.10 (3.11+ may have compatibility issues)
- **RAM**: Minimum 4GB, recommended 8GB+
- **Storage**: 2GB free space
- **OS**: Windows, macOS, or Linux

### Alternative Deployment Options

**Docker (For Production/Sharing)**
```bash
# Build and run with Docker
docker-compose up
# Access at http://localhost:8501
```

**Cloud Deployment (Advanced Users)**
- **Streamlit Cloud**: Free tier available (may have dependency issues)
- **Heroku**: Requires paid dyno for PyCaret
- **Railway**: Good alternative for cloud deployment
- **Google Colab**: Can run in browser (limited functionality)

## 📁 Project Structure

```
modela/
├── app.py                    # Main Streamlit app entry point
├── pages/                    # Streamlit pages
│   ├── 1_upload.py          # Dataset upload page
│   ├── 2_train.py           # Model training page
│   ├── 3_results.py         # Results dashboard
│   └── 4_projects.py        # Past projects list
├── utils/                    # Core utilities
│   ├── automl.py            # PyCaret wrapper functions
│   ├── firebase_client.py   # Local storage client
│   ├── preprocessing.py     # Data cleaning/validation
│   └── visualization.py     # Charts and metrics display
├── .github/                  # GitHub templates and workflows
│   ├── ISSUE_TEMPLATE/      # Issue templates
│   └── workflows/           # CI/CD pipeline
├── sample_data/              # Demo datasets (created by script)
├── data/                     # Local storage directory (auto-created)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Docker configuration
├── docker-compose.yml       # Docker Compose setup
├── setup.py                 # Automated setup script
├── test_modela.py           # Test suite
├── create_demo_data.py      # Demo data generator
├── README.md                # Project documentation
├── LICENSE                  # MIT License
├── CONTRIBUTING.md          # Contribution guidelines
├── CHANGELOG.md             # Version history
└── .streamlit/config.toml   # Streamlit configuration
```

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AutoML**: PyCaret
- **Authentication**: Simple local authentication
- **Storage**: Local file storage
- **Visualization**: Plotly
- **Deployment**: Streamlit Cloud (free tier)

## 📊 Supported Data Types

- **Classification**: Predicting categories (e.g., spam/not spam, customer churn)
- **Regression**: Predicting numerical values (e.g., house prices, sales forecasts)

## 🔧 Configuration

### Local Storage

The application automatically creates a `data/` directory to store:
- User accounts and authentication data
- Uploaded datasets
- Trained models and metadata

No additional configuration is required!

## 🎯 Demo Datasets

Modela comes with built-in demo datasets to help you get started quickly:

### Classification Datasets
- **Iris Flowers**: Classic 3-class classification problem
- **Wine Quality**: Wine classification based on chemical properties
- **Customer Churn**: Predict customer churn in business scenarios
- **Synthetic Classification**: Generated dataset for testing

### Regression Datasets
- **House Prices**: Predict house prices based on features
- **Synthetic Regression**: Generated dataset for testing

### Getting Demo Data
```bash
# Create sample datasets
python create_demo_data.py

# Test the installation
python test_modela.py
```

## 🎯 Usage Guide

### 1. Upload Dataset
- Click "Upload Dataset" in the sidebar
- Choose a CSV or Excel file (max 50MB)
- Preview your data and check the summary
- Save the dataset to local storage

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

- **Authentication**: Simple local authentication (email validation)
- **Guest Mode**: Anonymous users can try the app
- **Data Storage**: All data stored locally on your machine
- **Privacy**: No data is shared with third parties

## 💰 Cost Structure

- **Free Tier**: 50MB datasets, 1 training at a time
- **Streamlit Cloud**: Free tier (500MB RAM, public repos)
- **Local Storage**: Completely free (uses your local disk space)
- **Total Cost**: $0/month for MVP

## 🎯 Use Cases

### Business Applications
- **Customer Analytics**: Churn prediction, customer segmentation
- **Sales Forecasting**: Revenue prediction, demand planning
- **Risk Assessment**: Credit scoring, fraud detection
- **Quality Control**: Defect detection, process optimization

### Research & Education
- **Academic Research**: Quick model prototyping and validation
- **Data Science Education**: Hands-on learning without coding
- **Rapid Prototyping**: Test ideas before full implementation
- **Model Comparison**: Evaluate different algorithms quickly

### Personal Projects
- **Sports Analytics**: Game outcome prediction
- **Investment Analysis**: Stock price prediction
- **Health Monitoring**: Personal health metrics analysis
- **Home Automation**: Smart home optimization

## 🚀 Roadmap

### Version 1.1 (Q1 2025)
- [ ] REST API endpoints
- [ ] Model deployment to cloud platforms
- [ ] Advanced preprocessing options
- [ ] Ensemble model support

### Version 1.2 (Q2 2025)
- [ ] Hyperparameter tuning interface
- [ ] Time series forecasting
- [ ] Deep learning model support
- [ ] Batch prediction API

### Version 2.0 (Q3 2025)
- [ ] Multi-user collaboration
- [ ] Model versioning and management
- [ ] Advanced visualization dashboard
- [ ] Integration with popular ML platforms

## 🐛 Troubleshooting

### Common Issues

1. **PyCaret Installation Issues**
   - **Problem**: PyCaret fails to install on Python 3.11+
   - **Solution**: Use Python 3.8, 3.9, or 3.10
   - **Alternative**: Use Docker which has compatible Python version

2. **Dependency Conflicts**
   - **Problem**: Package version conflicts during installation
   - **Solution**: Create a virtual environment: `python -m venv modela_env`
   - **Activate**: `source modela_env/bin/activate` (Linux/Mac) or `modela_env\Scripts\activate` (Windows)

3. **Storage Error**
   - Ensure you have write permissions in the project directory
   - Check available disk space
   - Verify the `data/` directory can be created

4. **Model Training Fails**
   - Check dataset size (max 50MB)
   - Ensure target column has sufficient data
   - Try with a smaller dataset first

5. **Upload Issues**
   - Check file format (CSV/Excel only)
   - Verify file size (max 50MB)
   - Ensure file is not corrupted

6. **Streamlit Cloud Deployment Issues**
   - **Problem**: PyCaret compatibility with Python 3.13
   - **Solution**: Use local installation instead
   - **Alternative**: Try Railway or Heroku with Python 3.10

### Getting Help

- Check the [Streamlit documentation](https://docs.streamlit.io/)
- Review [PyCaret documentation](https://pycaret.readthedocs.io/)

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your contributions make Modela better for everyone.

### Quick Start for Contributors
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test them
4. Commit your changes: `git commit -m 'Add amazing feature'`
5. Push to the branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/youssof20/modela?style=social)
![GitHub forks](https://img.shields.io/github/forks/youssof20/modela?style=social)
![GitHub issues](https://img.shields.io/github/issues/youssof20/modela)
![GitHub pull requests](https://img.shields.io/github/issues-pr/youssof20/modela)

## 🏆 Recognition

Modela has been featured in:
- [ ] Data Science communities
- [ ] Open source showcases
- [ ] Educational resources
- [ ] Industry publications

## 📞 Support & Community

- **GitHub Issues**: [Report bugs and request features](https://github.com/youssof20/modela/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/youssof20/modela/discussions)
- **Email**: Contact the maintainer directly
- **Twitter**: Follow [@youssof20](https://twitter.com/youssof20) for updates

## 🙏 Acknowledgments

- **PyCaret**: For the amazing AutoML framework
- **Streamlit**: For the beautiful web framework
- **Contributors**: Thank you to all contributors who make this project better
- **Community**: Thanks to the open source community for inspiration and support

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ by [Youssof](https://github.com/youssof20) using Streamlit and PyCaret**

⭐ **Star this repository if you find it helpful!**
"# modela" 
