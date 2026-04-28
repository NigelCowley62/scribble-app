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
# FLOW FIELD
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

            if edge_strength > 0.25:
                theta = angle[iy, ix] + np.pi / 2

            local_chaos = chaos * (0.6 + tone * 1.2)
            local_chaos *= (0.85 + np.random.rand() * 0.3)

            if in_face(x, y):
                local_chaos *= 0.4

            theta += np.random.normal(0, local_chaos)

            step = 0.04 + (1 - tone) * 0.7

            x += np.cos(theta) * step
            y += np.sin(theta) * step

            if weight > 0.6:
                for _ in range(2):
                    jitter_x = x + np.random.normal(0, 0.3)
                    jitter_y = y + np.random.normal(0, 0.3)
                    path.append((jitter_x, jitter_y))

        return path

    seeds = []

    for _ in range(density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        edge_strength = edges[y, x] / 255

        prob = (tone ** 3.0) * 2.0 + edge_strength * 2.0

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

    edges = cv2.Canny(gray, 50, 130)

    # 🔥 MULTI-SCALE FLOW FIELD
    blur_small = cv2.GaussianBlur(gray, (3, 3), 0)
    blur_large = cv2.GaussianBlur(gray, (11, 11), 0)

    gx_small = cv2.Sobel(blur_small, cv2.CV_32F, 1, 0, ksize=3)
    gy_small = cv2.Sobel(blur_small, cv2.CV_32F, 0, 1, ksize=3)

    gx_large = cv2.Sobel(blur_large, cv2.CV_32F, 1, 0, ksize=5)
    gy_large = cv2.Sobel(blur_large, cv2.CV_32F, 0, 1, ksize=5)

    angle_small = np.arctan2(gy_small, gx_small)
    angle_large = np.arctan2(gy_large, gx_large)

    edge_norm = edges / 255.0
    angle = angle_small * edge_norm + angle_large * (1 - edge_norm)

    # generate points
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

    # 🔥 VISIT MAP
    visit_map = np.zeros_like(gray, dtype=np.float32)

    # -----------------------------
    # TSP WITH MOMENTUM + MEMORY
    # -----------------------------
    used = np.zeros(len(pts), dtype=bool)
    path = []

    current = np.random.randint(len(pts))
    path.append(tuple(pts[current]))
    used[current] = True

    direction = np.array([1.0, 0.0])
    k = 14

    for _ in range(len(pts) - 1):

        current_point = pts[current]

        dists = np.sum((pts - current_point)**2, axis=1).astype(float)
        dists[used] = np.inf

        nearest = np.argpartition(dists, k)[:k]
        nearest = nearest[np.isfinite(dists[nearest])]

        if len(nearest) == 0:
            break

        candidates = pts[nearest]

        local_dists = dists[nearest]
        local_dists[local_dists == 0] = 1e-6
        dist_score = np.exp(-local_dists / np.min(local_dists))

        vectors = candidates - current_point
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms == 0] = 1e-6

        unit_vectors = vectors / norms[:, None]

        dir_unit = direction / (np.linalg.norm(direction) + 1e-6)
        alignment = np.dot(unit_vectors, dir_unit)
        dir_score = (alignment + 1) / 2

        xs = np.clip(candidates[:,0].astype(int), 0, gray.shape[1]-1)
        ys = np.clip(candidates[:,1].astype(int), 0, gray.shape[0]-1)

        tones = 1 - gray[ys, xs] / 255
        visit_penalty = 1 / (1 + visit_map[ys, xs])

        tone_score = (tones ** 2.0) * visit_penalty

        scores = (
            dist_score * 0.25 +
            dir_score * 0.45 +
            tone_score * 0.30
        )

        scores = np.clip(scores, 1e-6, None)
        probs = scores / np.sum(scores)

        next_index = np.random.choice(nearest, p=probs)

        px = int(pts[next_index][0])
        py = int(pts[next_index][1])
        if 0 <= px < gray.shape[1] and 0 <= py < gray.shape[0]:
            visit_map[py, px] += 1.0

        new_direction = pts[next_index] - current_point
        direction = direction * 0.7 + new_direction * 0.3

        path.append(tuple(pts[next_index]))
        used[next_index] = True
        current = next_index

    pts = np.array(path)

    # 🔥 ADAPTIVE REDUCTION
    xs = np.clip(pts[:,0].astype(int), 0, gray.shape[1]-1)
    ys = np.clip(pts[:,1].astype(int), 0, gray.shape[0]-1)
    tones = 1 - gray[ys, xs] / 255

    mask = np.random.rand(len(pts)) < (0.4 + tones * 0.6)
    pts = pts[mask]

    # smoothing
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