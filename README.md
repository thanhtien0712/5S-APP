# OpEx Vision AI - Automated 5S Compliance System 🏭✨

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLO-World_v2-FFD700.svg?style=for-the-badge&logo=nodedotjs&logoColor=black)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4.svg?style=for-the-badge&logo=google&logoColor=white)

OpEx Vision AI is an advanced, automated workplace compliance auditing system designed for modern manufacturing and corporate environments. By leveraging state-of-the-art **Computer Vision (YOLO)** and **Generative AI (Google Gemini)**, it instantly evaluates a workspace against strict 5S (Sort, Set in order, Shine) standards.

## 🚀 Key Features

- **Real-Time Spatial Scanning**: Detects desktop objects like monitors, laptops, bottles, and stationery using YOLO-World zero-shot detection.
- **Smart Deduplication & Normalization**: Utilizes Gemini 2.5 to intelligently merge overlapping bounding boxes, correct misclassifications, and filter out noise (e.g., humans, shadows).
- **Automated 5S Scoring System**: Applies strict compliance quotas (e.g., max 2 monitors, max 2 water bottles) and penalizes unauthorized items (e.g., sharp objects, clutter, personal items).
- **Sci-Fi / Cyberpunk UI**: A beautifully crafted, responsive HUD interface with laser scanning animations and interactive audit reports.
- **Mobile-Ready**: Supports direct camera capture from mobile devices via local network or cloud deployment.

---

## 🛠️ Tech Stack

- **Backend**: FastAPI, Uvicorn, Python
- **Computer Vision**: Ultralytics YOLOv8 (YOLO-World), OpenCV
- **LLM / GenAI**: Google Generative AI (Gemini 2.5 Flash API)
- **Frontend**: HTML5, CSS3 (Glassmorphism, CSS Animations, Custom Variables), Vanilla JS

---

## ⚙️ Workflow Architecture

1. **Data Acquisition**: User uploads or captures an image of the workspace.
2. **Step 1 - Spatial Scanning**: YOLOv8 extracts bounding boxes and confidence scores for various office objects.
3. **Step 2 - AI Normalization**: Bounding box coordinates are overlaid on the original image and sent to Gemini 2.5 for context-aware deduplication and safety checks (e.g., classifying sharp objects like knives).
4. **Step 3 - Audit & Reporting**: Python calculates the final 5S score based on internal corporate quotas. Gemini drafts a concise, formatted HTML report detailing "Compliant Items" and "Action Required".

---

## 💻 Installation & Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/thanhtien0712/5S-APP.git
cd 5S-APP
```

### 2. Set up Virtual Environment & Dependencies
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Environment Variables
Create a file named `api_key.txt` or set an environment variable `GEMINI_API_KEY` with your Google Gemini API key:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```

### 4. Run the Application
You can use the provided batch script or run it directly:
```bash
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
Access the application at `http://localhost:8000` (or your machine's IP address on mobile devices).

---

## 🌐 Cloud Deployment (Render / HuggingFace)

For production deployment, use platforms that support heavy Python binaries (OpenCV + PyTorch).
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- *Note: Ensure you add `GEMINI_API_KEY` in the environment variables of your hosting provider.*

---

*Disclaimer: This project was built to demonstrate AI automation in Lean Manufacturing & Operational Excellence (OpEx).*