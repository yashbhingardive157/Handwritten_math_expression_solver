import numpy as np
import cv2 as cv
import classifier
import solver
import argparse

class CharacterDetector():
    def __init__(self, visualize=False):
        self.visualize = visualize

    def sorted_bbox(self, contours):
        
        #Returns the bounding boxe, sorted by their x-axis position.
        #Merges overlapping boundig boxes.
        
        rects = []
        rectsUsed = []
        for cnt in contours:
            rects.append(cv.boundingRect(cnt))
            rectsUsed.append(False)

        rects.sort(key=lambda x: x[0])
        acceptedRects = []
        xThr = 1
        for supIdx, supVal in enumerate(rects):
            if (rectsUsed[supIdx] == False):
                currxMin = supVal[0]
                currxMax = supVal[0] + supVal[2]
                curryMin = supVal[1]
                curryMax = supVal[1] + supVal[3]
                rectsUsed[supIdx] = True

                for subIdx, subVal in enumerate(rects[(supIdx + 1):], start=(supIdx + 1)):
                    candxMin = subVal[0]
                    candxMax = subVal[0] + subVal[2]
                    candyMin = subVal[1]
                    candyMax = subVal[1] + subVal[3]

                    if (candxMin <= currxMax + xThr):
                        currxMax = max(currxMax, candxMax)
                        curryMin = min(curryMin, candyMin)
                        curryMax = max(curryMax, candyMax)
                        rectsUsed[subIdx] = True
                    else:
                        break
                acceptedRects.append([currxMin, curryMin, currxMax - currxMin, curryMax - curryMin])
        return acceptedRects

    def detect(self, img):
        copy = img.copy()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray, 120, 255, cv.THRESH_BINARY_INV)
        canny = cv.Canny(thresh, 100, 200)
        contours, hierarchy = cv.findContours(canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        bounding_boxes = self.sorted_bbox(contours)
        crops = []
        for bbox in bounding_boxes:
            rect_x = bbox[0]
            rect_y = bbox[1]
            rect_w = bbox[2]
            rect_h = bbox[3]

            rect_area = rect_w * rect_h
            min_area = 150

            if rect_area > min_area:
                color = (0, 255, 0)
                cv.rectangle(copy, (int(rect_x), int(rect_y)), (int(rect_x + rect_w), int(rect_y + rect_h)), color, 2)
                current_crop = gray[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]
                crops.append(current_crop)
        if self.visualize:
            cv.imshow("Bounding Boxes", copy)
            cv.waitKey(0)
        return crops, copy, bounding_boxes

