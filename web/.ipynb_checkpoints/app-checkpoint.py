import os
from flask import Flask, render_template, request, send_from_directory
import cv2 as cv
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import sys

# Add utils, detector, classifier, solver imports
sys.path.append(os.path.abspath(".."))
import utils
import detector
import classifier
import solver

UPLOAD_FOLDER = 'uploads'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

detector = detector.CharacterDetector()
classifier = classifier.HandwritingClassifier()
solver = solver.Solver()

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

# Process the uploaded file and predict
@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    if file:
        filename = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filename)
        image = cv.imread(filename)
        
        crops, copy, bboxes = detector.detect(image)
        expression = ''
        for i, crop in enumerate(crops):
            prediction = classifier.run(crop)
            expression = expression + prediction + ' '
            cv.putText(copy, prediction, (bboxes[i][0] + 20, bboxes[i][1]-20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        # Save the processed image
        result_filename = file.filename[:file.filename.rindex('.')] + '_new' + file.filename[file.filename.rindex('.'): ]
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv.imwrite(result_path, copy)

        # Evaluate the expression
        result = solver.evaluate(expression)

        return render_template('upload.html', image_file_name=result_filename, expression=expression, result=result)
    
    return 'No file uploaded'

# Route to handle canvas drawing submission
@app.route('/draw', methods=['POST'])
def draw_image():
    if 'imageData' in request.form:
        image_data = request.form['imageData']  # Get the base64 image data
        # Strip the "data:image/png;base64," prefix if it exists
        image_data = image_data.split(',')[1]
        img_data = base64.b64decode(image_data)

        # Convert base64 image data to PIL image
        img = Image.open(BytesIO(img_data))
        # Save the drawn image to the server
        img_path = os.path.join(UPLOAD_FOLDER, 'drawn_expression.png')
        img.save(img_path)

        # Convert the image to an OpenCV format (for further processing)
        image = cv.imread(img_path)

        crops, copy, bboxes = detector.detect(image)
        expression = ''
        for i, crop in enumerate(crops):
            prediction = classifier.run(crop)
            expression = expression + prediction + ' '
            cv.putText(copy, prediction, (bboxes[i][0] + 20, bboxes[i][1]-20), cv.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

        # Save the processed image
        result_filename = 'drawn_expression_processed.png'
        result_path = os.path.join(UPLOAD_FOLDER, result_filename)
        cv.imwrite(result_path, copy)

        # Evaluate the expression
        result = solver.evaluate(expression)

        return render_template('upload.html', image_file_name=result_filename, expression=expression, result=result)
    
    return 'No drawing submitted'

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.debug = True
    app.run(debug=True)

