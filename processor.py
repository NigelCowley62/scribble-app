import cv2
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev

# -----------------------------
# FACE DETECTOR
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

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
# WATERMARK (SUBTLE)
# -----------------------------
def add_watermark(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return

    overlay = img.copy()
    h, w = img.shape[:2]

    text = "PREVIEW"
    font_scale = w / 800
    thickness = max(1, int(font_scale * 2))

    x = int(w * 0.15)
    y = int(h * 0.7)

    cv2.putText(
        overlay,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (200, 200, 200),
        thickness,
        cv2.LINE_AA
    )

    cv2.addWeighted(overlay, 0.25, img, 0.75, 0, img)
    cv2.imwrite(image_path, img)


# -----------------------------
# PREVIEW
# -----------------------------
def save_preview_from_path(path, png_path, size="A4", orientation="portrait"):

    pts = np.array(path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    data_w = max(max_x - min_x, 1e-6)
    data_h = max(max_y - min_y, 1e-6)

    base = {"A4":1400, "A3":1800, "A2":2200}.get(size, 2600)

    if orientation == "landscape":
        canvas_w = int(base * (data_w / data_h))
        canvas_h = base
    else:
        canvas_w = base
        canvas_h = int(base * (data_h / data_w))

    img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    scale = min(canvas_w / data_w, canvas_h / data_h)

    ox = (canvas_w - data_w * scale) / 2
    oy = (canvas_h - data_h * scale) / 2

    for i in range(1, len(pts)):
        x1 = int((pts[i-1][0] - min_x) * scale + ox)
        y1 = int((pts[i-1][1] - min_y) * scale + oy)
        x2 = int((pts[i][0] - min_x) * scale + ox)
        y2 = int((pts[i][1] - min_y) * scale + oy)

        cv2.line(img, (x1, y1), (x2, y2), (0,0,0), 1)

    cv2.imwrite(png_path, img)


# -----------------------------
# FLOW FIELD (TSP-OPTIMISED)
# -----------------------------
def generate_points(gray, angle, edges, density, chaos, faces):

    h, w = gray.shape
    points = []

    def in_face(x, y):
        for (fx, fy, fw, fh) in faces:
            if fx <= x <= fx + fw and fy <= y <= fy + fh:
                return True
        return False

    def trace_line(x, y):
        path = []

        ix0, iy0 = int(x), int(y)
        if ix0 < 0 or iy0 < 0 or ix0 >= w or iy0 >= h:
            return path

        tone0 = 1 - gray[iy0, ix0] / 255

        # 🔥 longer flowing strokes
        max_len = int(25 + tone0 * 80 + np.random.randint(0, 30))

        for _ in range(max_len):

            ix, iy = int(x), int(y)
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            tone = 1 - gray[iy, ix] / 255
            edge_strength = edges[iy, ix] / 255

            weight = (tone ** 3.0) * 2.0 + edge_strength * 1.6

            if in_face(x, y):
                weight *= 1.2

            weight = min(max(weight, 0), 1)

            theta = angle[iy, ix] + np.pi / 2

            # 🔥 smoother chaos (less noisy)
            local_chaos = chaos * (0.6 + tone * 1.2)
            local_chaos *= (0.85 + np.random.rand() * 0.3)

            if in_face(x, y):
                local_chaos *= 0.4

            theta += np.random.normal(0, local_chaos)

            # subtle wobble (kept small)
            theta += np.sin(ix * 0.05 + iy * 0.05) * 0.15

            # 🔥 slightly larger step = more flow
            step = 0.08 + (1 - tone) * 0.7

            x += np.cos(theta) * step
            y += np.sin(theta) * step

            # 🔥 light overlap (not scribbly)
            if weight > 0.6:
                repeats = int(1 + tone * 2)

                for _ in range(repeats):
                    jitter = np.random.normal(0, 0.2)

                    x += np.cos(theta + jitter) * step * 0.4
                    y += np.sin(theta + jitter) * step * 0.4

                    path.append((x, y))

        return path

    seeds = []

    for _ in range(density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        edge_strength = edges[y, x] / 255

        prob = (tone ** 2.8) * 2.0 + edge_strength * 1.4

        # slight randomness (not too much)
        prob *= (0.9 + np.random.rand() * 0.2)

        if in_face(x, y):
            prob *= 1.2

        prob = min(max(prob, 0), 1)

        if np.random.rand() < prob:
            seeds.append((x, y))

    if len(seeds) < 50:
        seeds += [(np.random.randint(0, w), np.random.randint(0, h)) for _ in range(200)]

    for s in seeds:
        line = trace_line(s[0], s[1])
        if len(line) > 10:
            for p in line:
                points.append((int(p[0]), int(p[1])))

    return points


# -----------------------------
# MAIN
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

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not load image")

    h, w = img.shape[:2]
    scale = 400 / max(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    gamma = 0.8
    gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    gray = np.clip(gray * 1.1, 0, 255).astype(np.uint8)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 130)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    angle = np.arctan2(gy, gx)

    # multi-layer (kept, but balanced)
    points = (
        generate_points(gray, angle, edges, int(density * 1.2), chaos * 1.1, faces) +
        generate_points(gray, angle, edges, int(density * 1.5), chaos * 0.9, faces) +
        generate_points(gray, angle, edges, int(density * 1.8), chaos * 0.5, faces)
    )

    pts = np.array(points)

    MAX_POINTS = min(int(density * 10), 22000)

    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[idx]

    # -----------------------------
    # TSP (SMOOTHER PATH)
    # -----------------------------
    used = np.zeros(len(pts), dtype=bool)
    path = []

    current = np.random.randint(len(pts))
    path.append(tuple(pts[current]))
    used[current] = True

    k = 12

    for _ in range(len(pts) - 1):

        dists = np.sum((pts - pts[current])**2, axis=1).astype(float)
        dists[used] = np.inf

        nearest = np.argpartition(dists, k)[:k]
        nearest = nearest[np.isfinite(dists[nearest])]

        if len(nearest) == 0:
            break

        local = dists[nearest]
        local[local == 0] = 1e-6

        # 🔥 smoother selection (less jumpy)
        weights = 1 / (local + 1e-6)
        weights /= np.sum(weights)

        next_index = np.random.choice(nearest, p=weights)

        path.append(tuple(pts[next_index]))
        used[next_index] = True
        current = next_index

    pts = np.array(path)

    if len(pts) > 6000:
        pts = pts[::2]

    try:
        tck, _ = splprep([pts[:,0], pts[:,1]], s=smoothness * 0.05)
        u = np.linspace(0, 1, len(pts))
        x, y = splev(u, tck)
        smooth_path = list(zip(x, y))
    except:
        smooth_path = path

    width_mm, height_mm = PRINT_SIZES.get(size, (210, 297))

    if orientation == "landscape":
        width_mm, height_mm = height_mm, width_mm

    pts = np.array(smooth_path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    scale = min(
        (width_mm - 20) / (max_x - min_x),
        (height_mm - 20) / (max_y - min_y)
    )

    scaled = [
        ((x - min_x) * scale + 10, (y - min_y) * scale + 10)
        for x, y in pts
    ]

    dwg = svgwrite.Drawing(output_path)
    dwg.viewbox(0, 0, width_mm, height_mm)

    dwg.add(dwg.polyline(
        scaled,
        stroke="black",
        fill="none",
        stroke_width=0.2
    ))

    dwg.save()

    save_preview_from_path(
        scaled,
        output_path.replace(".svg", ".png"),
        size=size,
        orientation=orientation
    )