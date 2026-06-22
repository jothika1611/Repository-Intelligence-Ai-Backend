# Backend

This is the backend service for the Repository Intelligence Agent. It is built with Python 3.12 and FastAPI.

## Requirements

- Python 3.12+
- Dependencies in `requirements.txt`

## Running Locally

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
   ```

## Docker

You can build and run this service using Docker:
```bash
docker build -t repository-backend .
docker run -p 8000:8000 repository-backend
```
