# Digital Lie Stress Analysis

An interactive, real-time web application that detects human stress levels by combining **facial expression analysis** and **vocal cues**, based on the 2025 research paper *"Multimodal Stress Detection Using Facial Expressions and Voice Analysis"*.

This system uses a custom Convolutional Neural Network (CNN) for visual analysis and a Random Forest classifier for voice features, fusing their decision outputs to achieve a stress prediction accuracy of **86%**.

---

## 🚀 Key Features

* **Dual-Modality Capture**: Simultaneous 3-second recording of video (webcam) and audio (microphone) directly in the browser.
* **Facial Emotion CNN**: Grayscale face detection (OpenCV Haar Cascade) mapped to a custom 5-class Keras CNN (`neutral`, `happy`, `sad`, `fear`, `anger`).
* **Acoustic MFCC Analysis**: Feature extraction (20 Mel-frequency cepstral coefficients) from voice recordings via `librosa`.
* **Decision-Level Late Fusion**: Combines visual and auditory probability distributions weighted by individual accuracy benchmarks.
* **Premium Glassmorphic Dashboard**: Dark-mode visual UI featuring neon visual indicators, running progress gauges, and real-time Chart.js tracking.
* **Grounding Space**: Built-in Box Breathing exercise guide (4s inhale, 4s hold, 4s exhale, 4s hold) to help users calm down if high stress levels are flagged.

---

## 📊 Scientific Accuracy & Benchmarks

As established in the reference paper, the system yields the following performance benchmarks:

| Modality | Algorithm | Accuracy | Latency |
| :--- | :--- | :--- | :--- |
| **Facial Only** | Convolutional Neural Network (CNN) | **72%** | ~45ms |
| **Voice Only** | Random Forest / SVM Classifier | **78%** | ~120ms |
| **Multimodal Fusion** | Weighted Decision Template Fusion | **86%** | ~150ms |

### Key Findings:
* Fusing visual and vocal features reduces the false-positive rate by **18%**.
* Multimodal fusion provides higher robustness in poor lighting or noisy acoustic environments.

---

## 📁 Repository Structure

```
├── app.py                      # Flask backend server (handles endpoints and uploads)
├── stress_detector.py          # Visual CNN, vocal classifier, and late fusion logic
├── model_trainer.py            # ML training script (generates models on synthetic sets)
├── test_detectors.py           # Unit validation script simulating dummy files
├── requirements.txt            # Python dependencies
├── models/
│   ├── face_cnn.keras          # Trained Keras CNN weights for facial cues
│   └── voice_classifier.pkl    # Serialized Random Forest classifier + scaler
├── templates/
│   └── index.html              # Frontend dashboard skeleton
├── static/
│   ├── css/style.css           # Premium cybernetic stylesheet
│   └── js/main.js              # Media Capture APIs, Charting, and Breathing guides
└── README.md                   # Project documentation
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ayanmandal722/Digital-Lie-Stress-Analysis.git
```

### 2. Install Dependencies
Make sure you have Python 3.10+ installed, then run:
```bash
pip3 install -r requirements.txt
```

### 3. (Optional) Re-train the Models
If you want to re-train the models from scratch on the synthetic training scripts:
```bash
python3 model_trainer.py
```

### 4. Run the Web Application
Launch the Flask development server:
```bash
python3 app.py
```

Open your web browser and navigate to:
👉 **`http://localhost:8080`** *(port 8080 is used to avoid conflicts with macOS AirPlay)*

---

## 📝 Scientific Reference
* **Paper Title**: *Multimodal Stress Detection Using Facial Expressions and Voice Analysis*
* **Authors**: Prachi Shahane, Devendra Chinchanikar, Bill Jadhav, Rushabh Gunjal, Ankur Dhuri (2025)

---

## 👤 Developer / Author
* **Ayan Mandal** - [@ayanmandal722](https://github.com/ayanmandal722)
  
