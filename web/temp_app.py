from flask import Flask, render_template, request, send_from_directory
import os
import cv2 as cv
import numpy as np
from PIL import Image
import base64
from io import BytesIO
import sys

# Add utils, detector, classifier, solver imports
sys.path.append(os.path.abspath(".."))
import utils
import detector
import classifier
import solver

UPLOAD_FOLDER = 'uploads'
DRAW_FOLDER = 'drawings'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['DRAW_FOLDER'] = DRAW_FOLDER

# Ensure folders exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(DRAW_FOLDER):
    os.makedirs(DRAW_FOLDER)

# Initialize models
detector = detector.CharacterDetector()
classifier = classifier.HandwritingClassifier()
solver = solver.Solver()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    if file:
        # Save uploaded image
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return process_image(filename, app.config['UPLOAD_FOLDER'])

    return 'No file uploaded'

@app.route('/draw', methods=['POST'])
def draw_image():
    if 'imageData' in request.form:
        # Get base64 image data from canvas
        image_data = request.form['imageData']
        image_data = image_data.split(',')[1]  # Remove the base64 header
        img_data = base64.b64decode(image_data)

        # Save the drawing as a JPEG in the drawings folder
        draw_filename = 'drawn_expression.jpeg'
        draw_path = os.path.join(app.config['DRAW_FOLDER'], draw_filename)
        img = Image.open(BytesIO(img_data)).convert('RGB')
        img.save(draw_path, 'JPEG')

        # Process the saved drawing
        return process_image(draw_path, app.config['DRAW_FOLDER'])

    return 'No drawing submitted'

def process_image(image_path, folder):
    """
    Process the given image: detect characters, classify them, and evaluate the math expression.
    """
    image = cv.imread(image_path)
    crops, copy, bboxes = detector.detect(image)
    expression = ''
    for i, crop in enumerate(crops):
        prediction = classifier.run(crop)
        expression += prediction + ' '
        cv.putText(copy, prediction, (bboxes[i][0] + 20, bboxes[i][1] - 20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

    # Save the processed image
    result_filename = os.path.basename(image_path).rsplit('.', 1)[0] + '_processed.jpeg'
    result_path = os.path.join(folder, result_filename)
    cv.imwrite(result_path, copy)

    # Evaluate the expression
    result = solver.evaluate(expression)

    return render_template('upload.html', image_file_name=result_filename, expression=expression, result=result)

@app.route('/uploads/<filename>')
def send_file(filename):
    # Serve images from both uploads and drawings folders
    if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], filename)):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    elif os.path.exists(os.path.join(app.config['DRAW_FOLDER'], filename)):
        return send_from_directory(app.config['DRAW_FOLDER'], filename)
    else:
        return "File not found", 404

if __name__ == '__main__':
    app.run(debug=True)

