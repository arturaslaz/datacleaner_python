# AI-Powered Data Cleaner

An intelligent data cleaning application that uses AI to automatically clean and process CSV/Excel datasets. Built with Streamlit, LangChain, and Google's Gemini LLM.

## Features

- Upload CSV/Excel files for cleaning
- AI-powered data analysis and cleaning suggestions
- Automated cleaning of common data issues:
  - Missing value handling
  - Duplicate removal
  - Outlier detection
  - Column name standardization
  - Data type conversion
  - Low-variance column removal
- Detailed cleaning report generation
- Clean data download in multiple formats

## Local Development

### Prerequisites

- Python 3.8+
- pip (Python package manager)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/datacleaner_python.git
cd datacleaner_python
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
export GOOGLE_API_KEY=your_gemini_api_key  # Required for Gemini LLM
```

### Running the App

Start the Streamlit app:
```bash
streamlit run app/main.py
```

The app will be available at http://localhost:8501

## Project Structure

```
datacleaner_python/
├── app/
│   ├── main.py              # Main Streamlit application
│   ├── components/          # UI components
│   ├── data_processing/     # Data cleaning functions
│   └── ai_agents/          # LangChain/LangGraph agents
├── tests/                   # Test files
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Docker Support

Build the Docker image:
```bash
docker build -t datacleaner .
```

Run the container:
```bash
docker run -p 8501:8501 datacleaner
```

## Deployment

The application is configured for deployment on Google Cloud Run. See the deployment documentation for detailed instructions.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

MIT License - see LICENSE file for details 