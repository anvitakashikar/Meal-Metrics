# 🍽️ Meal-Metrics

An AI-powered nutrition analysis app that uses ML models 
to analyze meals and provide detailed nutritional insights.

## Tech Stack
- Frontend: React + Vite
- Backend: FastAPI + Python 3.11
- Database: Firebase Firestore
- ML: TensorFlow, PyTorch, Transformers

## Folder Structure
- frontend/ — React app
- backend/ — FastAPI app
- ml/      — ML models and scripts

## How to Run

### Frontend
cd frontend
npm install
npm run dev

### Backend
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload

### ML
cd ml
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt