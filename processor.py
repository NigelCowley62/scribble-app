import cv2
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev


# -----------------------------
# PRINT SIZES (mm)
# -----------------------------
PRINT_SIZES = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "A1": (594, 841),
}


# -----------------------------
# PREVIEW GENERATION
# -----------------------------
def save_preview_from_path(path, png_path, size="A4", orientation="portrait"):
    import cv2
    import numpy as np

    pts = np.array(path)

    # get bounds
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    data_w = max_x - min_x
    data_h = max_y - min_y

    # avoid zero
    data_w = max(data_w, 1e-6)
    data_h = max(data_h, 1e-6)

    # define base size
    base = 400

    # compute canvas based on ACTUAL orientation
    if data_w > data_h:
        # landscape
        canvas_w = int(base * (data_w / data_h))
        canvas_h = base
    else:
        # portrait
        canvas_w = base
        canvas_h = int(base * (data_h / data_w))

    img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # scale to fit
    scale = min(canvas_w / data_w, canvas_h / data_h)

    offset_x = (canvas_w - data_w * scale) / 2
    offset_y = (canvas_h - data_h * scale) / 2

    # draw
    for i in range(1, len(pts)):
        x1 = int((pts[i-1][0] - min_x) * scale + offset_x)
        y1 = int((pts[i-1][1] - min_y) * scale + offset_y)
        x2 = int((pts[i][0] - min_x) * scale + offset_x)
        y2 = int((pts[i][1] - min_y) * scale + offset_y)

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 0), 1)

    cv2.imwrite(png_path, img)


# -----------------------------
# WATERMARK
# -----------------------------
def add_watermark(image_path):
    img = cv2.imread(image_path)
    overlay = img.copy()

    cv2.putText(
        overlay,
        "PREVIEW",
        (40, img.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (200, 200, 200),
        3,
        cv2.LINE_AA
    )

    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.imwrite(image_path, img)


# -----------------------------
# POINT GENERATION
# -----------------------------
def generate_points(gray, angle, edges, density, chaos):

    h, w = gray.shape
    points = []

    def trace_line(x, y):
        path = []

        for _ in range(100):
            ix, iy = int(x), int(y)

            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            tone = 1 - gray[iy, ix] / 255
            edge_strength = edges[iy, ix] / 255

            weight = (tone ** 2) * 0.7 + edge_strength * 0.6

            theta = angle[iy, ix] + np.pi / 2

            if edges[iy, ix] > 0:
                theta += np.random.normal(0, chaos * 0.3)
            else:
                theta += np.random.normal(0, chaos * (0.7 + (1 - weight)))

            step = 0.4 + (1 - weight)

            x += np.cos(theta) * step
            y += np.sin(theta) * step

        return path

    seeds = []

    for _ in range(density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        edge_strength = edges[y, x] / 255

        weight = (tone ** 2) * 0.7 + edge_strength * 0.6

        if weight > 0.02 and np.random.rand() < weight:
            seeds.append((x, y))

    if len(seeds) < 50:
        for _ in range(200):
            seeds.append((np.random.randint(0, w), np.random.randint(0, h)))

    for s in seeds:
        line = trace_line(s[0], s[1])
        if len(line) > 10:
            for p in line:
                points.append((int(p[0]), int(p[1])))

    return points


# -----------------------------
# MAIN PROCESS
# -----------------------------
def process_image_to_svg(
    input_path,
    output_path,
    density=1200,
    smoothness=5,
    chaos=0.2,
    size="A4",
    orientation="portrait"
):
    print("ORIENTATION:", orientation)
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not load image")

    img = cv2.resize(img, (400, 400))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 80, 160)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    angle = np.arctan2(gy, gx)

    # multi-pass
    points_light = generate_points(gray, angle, edges, int(density * 0.7), chaos * 1.2)
    points_dark  = generate_points(gray, angle, edges, int(density * 1.3), chaos * 0.6)

    points = points_light + points_dark
    points = points[::3]

    if len(points) < 50:
        raise ValueError("Not enough points generated")

    pts = np.array(points)

    # -----------------------------
    # TSP PATH
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
    # SMOOTH
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
    # SCALE TO PRINT SIZE
    # -----------------------------
    width_mm, height_mm = PRINT_SIZES.get(size, (210, 297))

    if orientation == "landscape":
        width_mm, height_mm = height_mm, width_mm

    margin = 10
    draw_w = width_mm - 2 * margin
    draw_h = height_mm - 2 * margin

    pts = np.array(smooth_path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    norm = (pts - [min_x, min_y]) / (np.array([max_x - min_x, max_y - min_y]) + 1e-6)

    scaled = []
    for x, y in norm:
        px = margin + x * draw_w
        py = margin + y * draw_h
        scaled.append((px, py))

    # -----------------------------
    # SAVE SVG
    # -----------------------------
    dwg = svgwrite.Drawing(
        output_path,
        size=(f"{width_mm}mm", f"{height_mm}mm")
    )

    dwg.add(dwg.polyline(
        scaled,
        stroke="black",
        fill="none",
        stroke_width=0.3
    ))

    dwg.save()

    # -----------------------------
    # SAVE PREVIEW
    # -----------------------------
    png_path = output_path.replace(".svg", ".png")

    save_preview_from_path(
    scaled,
    png_path,
    size=size,
    orientation=orientation
)