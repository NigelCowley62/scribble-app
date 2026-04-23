import cv2
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev


def process_image_to_svg(input_path, output_path, density=1200, smoothness=5, chaos=0.2):

    # -----------------------------
    # LOAD IMAGE
    # -----------------------------
    img = cv2.imread(input_path)

    if img is None:
        raise ValueError("Could not load image")

    img = cv2.resize(img, (400, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Improve tone contrast slightly
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # -----------------------------
    # FLOW FIELD
    # -----------------------------
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)

    angle = np.arctan2(gy, gx)

    h, w = gray.shape

    # -----------------------------
    # TRACE LINES
    # -----------------------------
    def trace_line(x, y):
        path = []

        for _ in range(100):
            ix, iy = int(x), int(y)

            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            # tone calculation
            tone = 1 - gray[iy, ix] / 255
            weight = tone ** 2

            # direction
            theta = angle[iy, ix] + np.pi / 2

            # controlled randomness
            theta += np.random.normal(0, chaos * (0.5 + (1 - weight)))

            # step size (tone controls density)
            step = 0.4 + (1 - weight) * 1.0

            x += np.cos(theta) * step
            y += np.sin(theta) * step

        return path

    # -----------------------------
    # GENERATE SEEDS (SAFE VERSION)
    # -----------------------------
    seeds = []

    for _ in range(density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        weight = tone ** 2

        # allow enough points through
        if weight > 0.01 and np.random.rand() < weight:
            seeds.append((x, y))

    # fallback if too few seeds
    if len(seeds) < 50:
        for _ in range(200):
            seeds.append((np.random.randint(0, w), np.random.randint(0, h)))

    # -----------------------------
    # GENERATE LINES
    # -----------------------------
    lines = []

    for s in seeds:
        line = trace_line(s[0], s[1])
        if len(line) > 10:
            lines.append(line)

    # -----------------------------
    # FLATTEN POINTS
    # -----------------------------
    points = []

    for line in lines:
        for p in line:
            points.append((int(p[0]), int(p[1])))

    # reduce size (performance)
    points = points[::3]

    if len(points) < 50:
        raise ValueError("Not enough points generated")

    pts = np.array(points)

    # -----------------------------
    # SIMPLE TSP PATH
    # -----------------------------
    used = np.zeros(len(pts), dtype=bool)

    path = []
    current = 0

    path.append(tuple(pts[current]))
    used[current] = True

    for _ in range(len(pts) - 1):
        current_point = pts[current]

        dists = np.sum((pts - current_point) ** 2, axis=1).astype(float)
        dists[used] = np.inf

        next_index = np.argmin(dists)

        path.append(tuple(pts[next_index]))
        used[next_index] = True
        current = next_index

    # -----------------------------
    # SMOOTH PATH
    # -----------------------------
    pts = np.array(path)

    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=smoothness)
        u = np.linspace(0, 1, len(pts))
        x, y = splev(u, tck)
        smooth_path = list(zip(x, y))
    except:
        smooth_path = path

    # -----------------------------
    # SAVE SVG
    # -----------------------------
    dwg = svgwrite.Drawing(output_path, size=(w, h))

    dwg.add(dwg.polyline(
        [(int(px), int(py)) for px, py in smooth_path],
        stroke="black",
        fill="none",
        stroke_width=0.5
    ))

    dwg.save()