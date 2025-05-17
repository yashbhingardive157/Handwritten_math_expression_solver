# Handwritten Math Expression Solver
application that recognizes and solves handwritten mathematical expressions using computer vision and deep learning.

## Features
- Accepts both image uploads and canvas drawings
- Recognizes digits (0-9) and operators (+, -, *, /, parentheses)
- Evaluates expressions following BODMAS rules
- Provides real-time results with processing visualization
- Includes undo/clear functionality for drawings

## Technologies
- Backend: Python (Flask, OpenCV, TensorFlow/Keras)
- Frontend: HTML5 Canvas, Bootstrap
- Machine Learning: Custom CNN model
- Image Processing: Contour detection and symbol segmentation

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run the application: `python app.py`
3. Access at: `http://localhost:5000`

## Future Improvements
- Support for advanced mathematical symbols
- Improved accuracy with expanded training data
- Mobile compatibility
- Real-time webcam input
