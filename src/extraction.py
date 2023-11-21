import cv2
import easyocr
import numpy as np

def maskLegendRegion(img, contour):
    # Create mask from contour
    mask = np.zeros_like(img)
    cv2.drawContours(mask, np.expand_dims(contour, axis=0), 0, color=(255,255,255), thickness=cv2.FILLED)
    # Apply mask to img
    masked_img = img & mask
    return masked_img

def extractLegends(image, legendcontour=None):
    # Apply mask if there is one
    if legendcontour is not None:
        image = maskLegendRegion(image, legendcontour)
    
    # Get Contours
    contours = generateContours(image)
    candidateContours = selectCandidateContours(contours, image.shape)

    # OCR on possible legends
    ocrcandidates = ocrContours(image, candidateContours)

    return ocrcandidates

def generateContours(img):
    # converting image into grayscale image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # AdaptiveThreshold to remove noise
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 4)

    # Edge Detection
    thresh_blur = cv2.GaussianBlur(thresh, (11, 11), 0)
    canny = cv2.Canny(thresh_blur, 0, 200)
    canny_dilate = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))

    # Finding contours for the detected edges.
    contours, hierarchy = cv2.findContours(canny_dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Do a further approx step to reduce the complexity of contours
    contours = [cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True) for c in contours]

    return contours

def selectCandidateContours(contours, image_dims):
    # Reducing selection to contours that are quaderlaterals of reasonable size.
    min_contour_size = image_dims[0]*image_dims[1]*0.000005 # 0.0005% of image area
    max_contour_size = image_dims[0]*image_dims[1]*0.0005 # 0.05% of image area
    valid_bounds = 0.50 # Percent +/- threshold from median

    candidate_areas = []
    candidate_rect_areas = []
    candidate_contours = []
    i = 0
    # list for storing names of shapes
    for c in contours:
        # here we are ignoring first counter because
        # findcontour function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(c, 0.01 * cv2.arcLength(c, True), True)

        # Only quadrilaterals shapes are valid
        if len(approx) != 4:
            continue

        # Discard shapes that are too large or small to be a valid legend
        area = cv2.contourArea(c)
        if area <= min_contour_size or area >= max_contour_size:
            continue

        # Discard shapes that are quads but not rectangular
        x,y,w,h = cv2.boundingRect(c)
        rect_area = w*h
        if rect_area > area*4:
            continue

        candidate_areas.append(area)
        candidate_rect_areas.append(rect_area)
        candidate_contours.append(c)

    # Calculate valid area threshold
    candidate_areas.sort()
    candidate_rect_areas.sort()

    median_area = np.median(candidate_areas)
    min_valid_area = median_area - median_area*valid_bounds
    max_valid_area = median_area + median_area*valid_bounds

    median_rect_area = np.median(candidate_rect_areas)
    min_valid_rect_area = median_rect_area - median_rect_area*valid_bounds
    max_valid_rect_area = median_rect_area + median_rect_area*valid_bounds

    # Narrow candidates further
    valid_contours = []
    for c in candidate_contours:
        # Discard shapes outside the valid contour areas
        area = cv2.contourArea(c)
        if area <= min_valid_rect_area or area >= max_valid_rect_area:
            continue

        valid_contours.append(c)

    return valid_contours

def ocrContours(img, contours):
    ocr_reader = easyocr.Reader(['en'])

    # Prep image for ocr
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Threshold the HSV image to get only black colors
    mask = cv2.inRange(hsv, np.array([0,0,10]), np.array([256,256,100]))
    # invert the mask to get black letters on white background
    res2 = cv2.bitwise_not(mask)

    # Individual legend items
    labels = {}

    i=0
    for c in contours:
        # find bounding box of shape
        x,y,w,h = cv2.boundingRect(c)

        ocr_result = ocr_reader.readtext(res2[y:y+h, x:x+w], paragraph=False)

        # Use the text as the label
        ocr_label = ''
        for seg in ocr_result:
            ocr_label += seg[1] + ', '
        ocr_label = ocr_label[:-2]

        # If no text is found then give a unique name
        if ocr_label == '':
            ocr_label = 'NO_OCR_RESULT_{}'.format(i)
            i += 1

        if ocr_label in labels:
            labels[ocr_label].append((ocr_result, c))
        else:
            labels[ocr_label] = [(ocr_result, c)]
    
    # Convert to json format
    features = []
    for f in labels:
        features.append({'label' : f, 'points' : np.squeeze(labels[f][0][1])})
    return features

