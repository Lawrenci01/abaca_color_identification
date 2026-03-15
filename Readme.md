# Abaca Color Scanner (v2)

An AI-powered system for grading abaca fiber color using the RHS (Royal Horticultural Society) color standard. This project employs a hybrid approach combining Computer Vision (CV) and Machine Learning (ML) to achieve high accuracy (90%+) in predicting RHS color codes (876 classes).

## 🚀 Key Features
- **Quad MLP Ensemble**: A combination of four diverse Multi-Layer Perceptron architectures to robustly classify fiber features.
- **Hybrid Scoring**: Uses a weighted combination of Delta-E color distance (75%) and MLP ensemble probabilities (25%) for final grading.
- **Foreground Extraction**: Intelligent segmentation using Otsu thresholding and morphological cleanup to isolate fiber from the scanner background.
- **Data Augmentation**: Robust pipeline generating ~870,000 augmented images to handle various lighting and texture conditions.
- **Real-time Serving**: Flask-based API with hot-reloading capabilities for model updates.
- **Cloud Integration**: Supabase integration for scan history, user management, and verified data collection.

## 🛠 Tech Stack
- **Languages**: Python 3.10+
- **ML Frameworks**: scikit-learn
- **Image Processing**: OpenCV, Pillow, scikit-image
- **Database**: Supabase
- **Deployment**: Docker, Hugging Face Spaces

## 📂 Project Structure
- `app.py`: Main Flask application and API.
- `features.py`: Core feature extraction (248-dim) and prediction logic.
- `train_model.py`: Training script for the Quad MLP ensemble.
- `evaluate.py`: Model evaluation and reporting.
- `build_rhs_csv.py`: Extracts reference colors from real photos of RHS cards.
- `process_real_photos.py`: Extracts texture swatches for training data.
- `augment_dataset.py`: Generates the augmented training dataset.
- `segment.py`: Fiber segmentation logic.
- `db.py`: Supabase database interface.
- `run_pipeline.py`: Orchestrator for the entire development/training lifecycle.

## 🏁 Getting Started

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Training the Pipeline
You can run the entire pipeline from color extraction to evaluation using the orchestrator:
```bash
python run_pipeline.py --all
```
This will:
1. Build the RHS color reference (`build_rhs_csv.py`).
2. Extract texture swatches (`process_real_photos.py`).
3. Augment the dataset (`augment_dataset.py`).
4. Train the Quad MLP ensemble (`train_model.py`).
5. Evaluate the results and generate a report (`evaluate.py`).

### 3. Running the App
```bash
python app.py
```
The app will be available at `http://localhost:7860`.

## 🧠 Architecture Details
- **Feature Vector (248-dim)**: Includes Lab/RGB/HSV histograms, color moments, LBP (Local Binary Patterns), Gabor filters, and spatial Lab regions.
- **Ensemble Weights**: `[0.28, 0.27, 0.22, 0.23]` for models A, B, C, and D respectively.
- **Calibrated Scoring**: Match scores are calibrated using an exponential decay function based on Delta-E, where `dE < 3` is a "Strong Match".

## ☁️ Deployment
The project is configured for deployment to Hugging Face Spaces via Docker:
```bash
python deploy_to_hf.py --token <YOUR_HF_TOKEN>
```

## ⚖️ License
MIT License