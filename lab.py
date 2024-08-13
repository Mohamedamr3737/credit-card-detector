import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import os
import cv2


def load_img(image_number):
    """Load and return an image based on the provided image number."""
    # Define a list of image file paths
    image_paths = [
        r"01 - Straightforward.jpg",
        r"02 - You can do it.jpg",
        r"03 - Should be okay.jpg",
        r"04 - Still ok, I hope.jpg",
        r"05 - Looks cool, hope it runs cool too.jpg",
        r"06 - Hatetlewe7 hatlewe7.jpg",
        r"07 - Hatet3eweg hat3eweg.jpg",
        r"08 - Ew3a soba3ak ya3am.jpg",
        r"09 - El spero spathis we23et 3aaaa.jpg",
        r"10 - Mal7 w Felfel.jpg",
        r"11 - Ya setty ew3i.jpg",
        r"11 - Ya setty ew3i.jpg",
        r"13 - Matozbot el camera ya Kimo.jpg",
        r"14 - 2el noor 2ata3.jpg",
        r"15 - Compresso Espresso.jpg",
        r"16 - Sheel el kart yastaaaa.jpg"
    ]

    # Check if the provided image_number is valid
    if 1 <= image_number <= len(image_paths):
        # Load the image corresponding to the image_number
        img_path = image_paths[image_number - 1]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        return img


import cv2
import numpy as np


def create_vertical_edge_mask(image):
    """Create a mask that shows horizontal edges with at least 3 contiguous pixels."""
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    #edges = cv2.Canny(image, 50, 80)
    edges = image
    # Create a mask to store the result
    mask = np.zeros_like(edges)

    # Iterate over each row to detect horizontal lines with at least 3 contiguous pixels
    rows, cols = edges.shape
    for r in range(rows):
        # Find contours in the row
        row_edges = edges[r, :]
        start = -1
        length = 0
        for c in range(cols):
            if row_edges[c] == 255:
                if start == -1:
                    start = c
                length += 1
            else:
                if length >= 3:
                    mask[r, start:start + length] = 255
                start = -1
                length = 0
        # Check at the end of the row
        if length >= 3:
            mask[r, start:start + length] = 255

    # Apply the mask to highlight the edges vertically
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    return masked_image

def detect_perpendicular_lines(image, vertical_x, threshold=50):
    """
    Detects intersections of a vertical line with perpendicular lines in the image.

    Parameters:
    image (np.ndarray): Input image.
    vertical_x (int): X-coordinate of the vertical line.
    threshold (int): Threshold for detecting lines using Hough Lines.

    Returns:
    list of tuples: List of y-coordinates where the vertical line intersects with perpendicular lines.
    """
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Detect lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold, minLineLength=30, maxLineGap=10)

    if lines is None:
        return []

    intersections = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            if abs(x1 - x2) > abs(y1 - y2):  # Check if the line is mostly vertical
                if vertical_x >= min(x1, x2) and vertical_x <= max(x1, x2):
                    # Compute the intersection point with the vertical line
                    # The x-coordinate of the vertical line is fixed at vertical_x
                    if x1 != x2:  # Avoid division by zero
                        slope = (y2 - y1) / (x2 - x1)
                        intercept = y1 - slope * x1
                        intersection_y = slope * vertical_x + intercept
                        intersections.append(int(intersection_y))

    # Sort intersections and remove duplicates
    intersections = sorted(set(intersections))

    return intersections


def crop_image_at_intersections(image, vertical_x, threshold=50):
    """
    Crops the image from the topmost to the bottommost intersection of a vertical line with perpendicular lines.

    Parameters:
    image (np.ndarray): Input image.
    vertical_x (int): X-coordinate of the vertical line.
    threshold (int): Threshold for detecting lines using Hough Lines.

    Returns:
    np.ndarray: Cropped image.
    """
    intersections = detect_perpendicular_lines(image, vertical_x, threshold)

    if len(intersections) < 2:
        print("Not enough intersections detected to perform cropping.")
        return image

    # Get the topmost and bottommost intersections
    top_y = min(intersections)
    bottom_y = max(intersections)

    # Crop the image from top to bottom intersection
    cropped_image = image[top_y:bottom_y, :]

    return cropped_image


# Example usage
# Load an example image

# Display results



def rotate_line(x1, y1, x2, y2, angle_degrees):

    # Convert angle from degrees to radians
    angle_radians = np.deg2rad(angle_degrees)

    # Rotation matrix
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians), np.cos(angle_radians)]
    ])

    # Original coordinates
    start_point = np.array([x1, y1])
    end_point = np.array([x2, y2])

    # Rotate the points
    rotated_start_point = rotation_matrix @ start_point
    rotated_end_point = rotation_matrix @ end_point

    # Return the new coordinates
    return rotated_start_point[0], rotated_start_point[1], rotated_end_point[0], rotated_end_point[1]
def show_lines_images(image, lines):
    """Show individual images of each detected line using matplotlib."""
    if lines is None:
        print("No lines detected.")
        return

    # Convert the image to RGB (if it is in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    line_image = np.zeros_like(image_rgb)
    # Iterate over each detected line
    for idx,line in enumerate(lines):
        #print(enumerate(lines))
        # Create a black image of the same size as the original image


        # Draw the current line on the black image
        x1, y1, x2, y2 = line[0]
        cv2.line(line_image, (x1, y1), (x2, y2), (255, 255, 255), 2)  # White line with thickness of 2
        length = np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        # Display the image with the line using matplotlib
    return line_image


def detect_longest_line(lines, max_length=0):
    """Return the longest line from a list of lines, optionally ignoring a line with max_length."""
    if lines is None or len(lines) == 0:
        return None, 0  # Return None and 0 if no lines are detected

    longest_line = None
    max_length_found = -1  # Initialize to a very small value to find the maximum length

    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)

        # If max_length is given and matches the current length, skip this line
        if max_length and length >= max_length:
            continue

        # Update the longest line if the current length is greater
        if length > max_length_found:
            max_length_found = length
            longest_line = line

    # If no valid longest line found, return None
    if longest_line is None:
        return None, 0

    return longest_line, max_length_found

def get_angle(x1, y1, x2, y2):
    """ Calculate the angle of the line segment (x1, y1) to (x2, y2). """
    return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

def find_perpendicular_line(longest_line, lines, angle_threshold=30):
    """ Check if there is a line perpendicular to the longest line. """
    if longest_line is None:
        return None

    x1, y1, x2, y2 = longest_line[0]
    longest_line_angle = get_angle(x1, y1, x2, y2)
    length = np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
    #print(longest_line_angle,length)
    for line in lines:
        x1_l, y1_l, x2_l, y2_l = line[0]
        angle = get_angle(x1_l, y1_l, x2_l, y2_l)
        length = np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)
        #print(angle,length)
        angle_diff = abs(longest_line_angle - angle)
        # Check if the line is perpendicular within the given angle threshold
        if abs(angle_diff - 90) <= angle_threshold or abs(abs(angle_diff - 180) - 90) <= angle_threshold:
            return line

    return None

def rotate_image(image, angle):
    """ Rotate the image around its center by the given angle. """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated_image

def detect_horizontal_edges(image, threshold=50, minLineLength=100, maxLineGap=10, tolerance=5):
    """
    Detects horizontal edges in an image using Hough Line Transform.

    Parameters:
    image (np.ndarray): Input grayscale image.
    threshold (int): Threshold for the Hough Line Transform.
    minLineLength (int): Minimum length of a line to be considered.
    maxLineGap (int): Maximum allowed gap between line segments to be connected.
    tolerance (int): Tolerance for considering a line as horizontal (in pixels).

    Returns:
    np.ndarray: A 3D numpy array of detected horizontal lines with shape (N, 1, 4).
    """
    # Convert image to grayscale if it's not already

    gray = image

    # Perform Hough Line Transform
    lines = cv2.HoughLinesP(gray, 1, np.pi / 180, threshold, minLineLength, maxLineGap)

    if lines is None:
        return np.array([])  # No lines detected

    # Convert lines to a 2D array for easier manipulation
    lines = np.squeeze(lines)  # Shape: (N, 4)

    # Filter out horizontal lines
    horizontal_lines = []
    for x1, y1, x2, y2 in lines:
        # Check if the line is horizontal within the given tolerance
        if abs(y1 - y2) <= tolerance:
            horizontal_lines.append([x1, y1, x2, y2])
    horizontal_lines =np.array(horizontal_lines).reshape(-1, 1, 4)
    for line in horizontal_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # Convert list of horizontal lines to a numpy array
    return image

def process_image_for_horizontal_longest_line(image,OG):
    """ Process the image to rotate if the longest line has a perpendicular line. """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(gray_image, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
    max_y_coords = np.maximum(lines[:, 0, 1], lines[:, 0, 3])

    # Sort lines based on the highest y-coordinate in descending order
    sorted = np.argsort(max_y_coords)[::-1]
    lines=lines[sorted]
    #show_lines_images(image,lines)

    #show_lines_images(image,lines)
    # Find the longest line
    longest_line,maxL = detect_longest_line(lines)
    linez = [longest_line, longest_line]
    # Find a perpendicular line
    #show_lines_images(image, linez)
    #lines = lines.tolist()
    for i in range(len(lines)):
        perpendicular_line = find_perpendicular_line(longest_line, lines)
        #print(i,maxL)
        if perpendicular_line is not None:
            # Calculate the angle to rotate the image
            x1, y1, x2, y2 = longest_line[0]
            longest_line_angle = get_angle(x1, y1, x2, y2)
            linez=[longest_line,longest_line]
            # We want the longest line to be horizontal (0 degrees)
            rotation_angle = longest_line_angle
            #show_lines_images(image,linez)
            # Rotate the image

            rotated_image = rotate_image(gray_image, -rotation_angle)
            #x1,y1,x2,y2=longest_line[0]
            height,width=OG.shape
            #OG = OG[0:max(y1,y2),0:width]
            print(rotation_angle)
            OG = rotate_image(OG, -rotation_angle)

            # Apply Hough Line Transform
            lines = cv2.HoughLinesP(rotated_image, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)
            # show_lines_images(image,lines)
            # Find the longest line
            maxL = 0
            longest_line, maxL = detect_longest_line(lines)

            # Find a perpendicular line

            # lines = lines.tolist()
            filtered_lines = []
            for i in range(len(lines)):
                perpendicular_line = find_perpendicular_line(longest_line, lines)
                # print(i,maxL)

                if perpendicular_line is not None:
                    # Calculate the angle to rotate the image
                    filtered_lines.append(perpendicular_line)
                    filtered_lines.append(longest_line)
                    print(filtered_lines)
                    continue
                    #show_image(OG,'xz')
                    #OG = OG[0:int(max(y1, y2)), 0:width]
                    # We want the longest line to be horizontal (0 degrees)
                    #return rotated_image, OG

                else:
                    # Return the original image if no perpendicular line is found
                    longest_line, maxL = detect_longest_line(lines, maxL)
                    continue

            #print(y1, y2)
            #x1,y1,x2,y2=rotate_line(x1,y1,x2,y2,-longest_line_angle)
            #print(y1,y2)
            #OG = OG[0:int(max(y1, y2)), 0:width]
            rotated_image=show_lines_images(rotated_image,filtered_lines)
            return rotated_image, OG
        else:
            # Return the original image if no perpendicular line is found
            longest_line, maxL = detect_longest_line(lines,maxL)
            continue
    print('oh no')
    return gray_image,OG


def detect_and_draw_longest_lines(canny_image):
    # Apply Hough Line Transform
    lines = cv2.HoughLinesP(canny_image,1, np.pi / 180, 100, minLineLength=50, maxLineGap=10)

    if lines is None:
        return canny_image

    # Define an angle threshold for grouping lines
    angle_threshold = 5  # degrees

    def get_angle(x1, y1, x2, y2):
        return np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi

    def calculate_length(x1, y1, x2, y2):
        return np.sqrt((x2 - x1) * 2 + (y2 - y1) * 2)

    # Group lines by angle
    grouped_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = get_angle(x1, y1, x2, y2)

        found_group = False
        for group in grouped_lines:
            if abs(group['angle'] - angle) < angle_threshold:
                group['lines'].append(line)
                found_group = True
                break

        if not found_group:
            grouped_lines.append({'angle': angle, 'lines': [line]})

    # Find the longest line in each group
    longest_lines = []
    for group in grouped_lines:
        max_length = 0
        longest_line = None
        for line in group['lines']:
            x1, y1, x2, y2 = line[0]
            length = calculate_length(x1, y1, x2, y2)
            if length > max_length:
                max_length = length
                longest_line = line
        if longest_line is not None:
            longest_lines.append({'angle': group['angle'], 'line': longest_line})

    # Collect the angles of the longest lines
    longest_line_angles = [group['angle'] for group in longest_lines]

    # Filter lines to keep only those with the same angle as the longest line
    filtered_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = get_angle(x1, y1, x2, y2)
        if any(abs(angle - longest_angle) < angle_threshold for longest_angle in longest_line_angles):
            filtered_lines.append(line)

    # Draw the filtered lines on the image
    height,width=canny_image.shape
    color_image = np.zeros((height, width, 3), dtype=np.uint8)
    for line in filtered_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return color_image


def generate_templates(font_path, output_dir, font_size=50):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    font = ImageFont.truetype(font_path, font_size)

    for char in range(48, 58):  # ASCII printable characters
        char_str = chr(char)
        image = Image.new('L', (font_size, font_size), color=255)  # 'L' for grayscale
        draw = ImageDraw.Draw(image)

        # Calculate text size
        bbox = draw.textbbox((0, 0), char_str, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        # Draw the text centered
        draw.text(((font_size - text_width) / 2, (font_size - text_height) / 2), char_str, font=font, fill=0)

        image.save(os.path.join(output_dir, f'{char_str}.png'))


def match_template(img, template_dir):
    image = img
    for template_name in os.listdir(template_dir):
        template_path = os.path.join(template_dir, template_name)
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_name = template_name.split('.')[0]

        result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
        threshold = 0.1
        loc = np.where(result >= threshold)

        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + template.shape[1], pt[1] + template.shape[0]), (0, 255, 0), 2)
            print(f'Match found for {template_name} at position {pt}')

    cv2.imwrite('result.png', image)
    cv2.imshow('Matching Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def show_image(image, title):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.show()
def opening(image):
    kernel = np.ones((3, 3), np.uint8)

    # Apply opening operation
    opened_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

    # Display results
    cv2.imshow('Original Image', image)
    cv2.imshow('Opened Image', opened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def closing(image):

    kernel = np.ones((3, 3), np.uint8)
    # Apply opening operation
    # Apply closing operation
    closed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    return closed_image
# Display results
def tagroba(image):
    height,width=image.shape
    #imagez=image
    for y in range(height):
        for x in range(width):
            if image[y, x] >245:
                image[y,x] =255
            else:
                image[y,x]=0
def erosion(image, kernel_size=3, iterations=1):

    # Define the structuring element (kernel)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(image, kernel, iterations=iterations)

    return eroded_image
def dilation(image, kernel_size=3):
    """
    Apply dilation to a binary image using a specified structuring element.

    Parameters:
        image (numpy.ndarray): Input binary image (must be binary with 0 and 255 values).
        kernel_size (int): Size of the structuring element (must be odd).

    Returns:
        numpy.ndarray: Dilated image.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")

    # Create a 3x3 structuring element
    structuring_element = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply dilation
    dilated_image = cv2.dilate(image, structuring_element, iterations=1)

    return dilated_image
def conV(img):
    nimg=img
    height, width = img.shape
    for y in range(2,height-2):
        for x in range(2,width-2):
            if img[y-1,x]==255  and img[y+1,x]==255 :
                nimg[y,x]=255
    return nimg
def conH(img):
    nimg=img
    height,width=img.shape
    for y in range(2,height-2):
        for x in range(2,width-2):
            if img[y,x-1]==255  and img[y,x+1]==255 :
                nimg[y,x]=255
    return nimg
def sharp(image, alpha=1):

    # Define a sharpening kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + alpha, -1],
                       [0, -1, 0]], dtype=np.float32)

    # Apply the sharpening filter
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image
def padder(img):
    clr = int(img[5, 5])
    img = cv2.copyMakeBorder(
        img,
        50,
        50,
        50,
        50,
        cv2.BORDER_CONSTANT,
        value=[clr]  # Color of padding (black in this case)
    )
  # White pixel or y- bottom<-10
    return img
def sobel(img,threshold1=50,threshold2=60):
    # @title Interactive Sobel+Canny { run: "auto", display-mode: "both" }
    ksizex = 3 # @param {type:"slider", min:1, max:13, step:2}
    ksizey = 1 # @param {type:"slider", min:1, max:13, step:2}
    scalex = 1 # @param {type:"slider", min:1, max:10, step:1}
    scaley = 2 # @param {type:"slider", min:1, max:10, step:1}
    deltax = 0 # @param {type:"slider", min:0, max:255, step:1}
    deltay = 0 # @param {type:"slider", min:0, max:255, step:1}
 # @param {type:"slider", min:1, max:255, step:1}
    L2gradient = True # @param {type:"boolean"}
    dx = cv2.Sobel(img,cv2.CV_16S,1,0,None,ksizex,scalex,deltax)
    dy = cv2.Sobel(img,cv2.CV_16S,0,1,None,ksizey,scaley,deltay)
    sobelcanny = cv2.Canny(dx,dy,threshold1,threshold2,None,L2gradient)
    #show_image(sobelcanny,'5')
    return sobelcanny
def rotato(sobelcanny):
    edges = sobelcanny
    top = -1
    bottom = -1
    left = -1
    right = -1

    # Get image dimensions
    height, width = edges.shape

    # Find the highest white pixel in each row (for horizontal edge)
    for y in range(height):
        for x in range(width):
            if edges[y, x] == 255:  # White pixel or y- bottom<-10
                if top == -1:
                    top = y
                if y - bottom > 10:
                    bottom = y
                    bottomx = x
                if (left == -1 or x <= left):
                    left = x
                    lefty = y
                if right == -1 or x >= right:
                    right = x
                    righty = y
    print(f"Top: {top} , Bottom: {bottom},{bottomx} ,Left: {left},{lefty}, Right: {right}")
    distL = np.sqrt(pow((lefty - bottom), 2) + pow((left - bottomx), 2))
    distR = np.sqrt(pow((righty - bottom), 2) + pow((right - bottomx), 2))
    coef = 1

    if abs(bottomx - left) > 30 and abs(bottom - lefty) > 30:
        if distR > distL:
            lefty = righty
            left = right
            coef = -1
            print("help")
        print("rot")
        tangle = (bottom - lefty) / (bottomx - left)
        angle = np.degrees(np.arctan(tangle))
        w, h = sobelcanny.shape
        print(angle)
        r = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        abs_cos = abs(r[0, 0])
        abs_sin = abs(r[0, 1])

        # Calculate the new width and height of the image
        new_w = int(h * abs_sin + w * abs_cos)
        new_h = int(h * abs_cos + w * abs_sin)

        # Adjust the rotation matrix to take into account the translation
        r[0, 2] += (new_w / 2) - w / 2
        r[1, 2] += (new_h / 2) - y / 2
        sobelcanny = cv2.warpAffine(sobelcanny, r, (new_w, new_h), flags=cv2.INTER_LINEAR)
        #show_image(sobelcanny, "aftr rot")
    return sobelcanny