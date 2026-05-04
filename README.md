# Assignment-Silver
Final Silver Badge Assignment

### Prerequisites
- Python 3.12 or higher
- OpenAI API Key (for Gen-AI features)

### API Key Configuration
Set the environment variable before running the app:
```bash
export OPENAI_API_KEY="write-api-key"
```

### Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```

2. **Activate the virtual environment:**
   ```bash
   .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true --browser.gatherUsageStats false
   ```

5. **Access the application:**
   - Open your web browser and go to: http://localhost:8501
   - The app will be running on port 8501
