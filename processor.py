import cv2
import numpy as np
import svgwrite

def process_image_to_svg(input_path, output_path):

    img = cv2.imread(input_path)

    # Resize for speed (important)
    img = cv2.resize(img, (400, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Compute gradients (this gives direction)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)

    angle = np.arctan2(gy, gx)

    h, w = gray.shape

    def trace_line(x, y):
        path = []

        for _ in range(120):
            ix, iy = int(x), int(y)

            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            # follow contour direction (perpendicular to gradient)
            theta = angle[iy, ix] + np.pi / 2

            # add slight randomness (makes it organic)
            theta += np.random.normal(0, 0.2)

            # darker areas = slower movement = more detail
            weight = 1 - gray[iy, ix] / 255
            step = 0.5 + weight

            x += np.cos(theta) * step
            y += np.sin(theta) * step

        return path

    # Generate starting points (biased to dark areas)
    seeds = []
    for _ in range(1500):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        if np.random.rand() < (1 - gray[y, x] / 255):
            seeds.append((x, y))

    lines = []

    for s in seeds:
        line = trace_line(s[0], s[1])
        if len(line) > 20:
            lines.append(line)

    # Create SVG
    dwg = svgwrite.Drawing(output_path, size=(w, h))

    for line in lines:
        dwg.add(dwg.polyline(
            [(int(x), int(y)) for x, y in line],
            stroke="black",
            fill="none",
            stroke_width=0.5
        ))

    dwg.save()