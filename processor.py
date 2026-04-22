import cv2
import numpy as np
import svgwrite

def process_image_to_svg(input_path, output_path):

    img = cv2.imread(input_path)

    # resize (IMPORTANT for speed)
    img = cv2.resize(img, (400, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    points = np.column_stack(np.where(edges > 0))

    # random shuffle for scribble effect
    np.random.shuffle(points)

    path = [(int(p[1]), int(p[0])) for p in points]

    dwg = svgwrite.Drawing(output_path, size=(400, 400))
    dwg.add(dwg.polyline(path, stroke="black", fill="none", stroke_width=1))

    dwg.save()