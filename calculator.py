import detector
import classifier
import solver
import cv2 as cv

class Calculator:
    def __init__(self, visualize=False):
        self.solver = solver.Solver()
        self.detector = detector.CharacterDetector(visualize)
        self.classifier = classifier.HandwritingClassifier()

    def calculate(self, img):
        crops, _, _ = self.detector.detect(img)
        expression = ''.join(self.classifier.run(crop) + ' ' for crop in crops)
        result = self.solver.evaluate(expression)
        print(f'Expression: {expression}')
        print(f'Result: {result}')

def main():
    img_path = 'examples/test.jpeg'
    visualize = False
    img = cv.imread(img_path)
    if img is None:
        print("file path is not correct")
        return
    calculator = Calculator(visualize=visualize)
    calculator.calculate(img)

if __name__ == '__main__':
    main()
