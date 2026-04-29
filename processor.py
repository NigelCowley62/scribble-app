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

    font_scale = w / 800
    thickness = max(1, int(font_scale * 2))

    cv2.putText(
        overlay,
        "PREVIEW",
        (int(w * 0.15), int(h * 0.7)),
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
# FLOW FIELD (TONALLY AWARE)
# -----------------------------
def generate_points(gray, angle, edges, density, chaos, faces):

    h, w = gray.shape
    points = []

    def trace_line(x, y):
        path = []

        tone0 = 1 - gray[int(y), int(x)] / 255
        max_len = int(25 + tone0 * 80 + np.random.randint(0, 30))

        for _ in range(max_len):

            ix, iy = int(x), int(y)
            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            tone = 1 - gray[iy, ix] / 255
            edge_strength = edges[iy, ix] / 255

            theta = angle[iy, ix] + np.pi / 2

            # edge anchoring
            if edge_strength > 0.25:
                theta = angle[iy, ix] + np.pi / 2

            local_chaos = chaos * (0.6 + tone * 1.2)
            theta += np.random.normal(0, local_chaos)

            step = 0.03 + (1 - tone) * 0.6

            x += np.cos(theta) * step
            y += np.sin(theta) * step

            # dark density boost
            if tone > 0.6:
                for _ in range(2):
                    path.append((
                        x + np.random.normal(0, 0.3),
                        y + np.random.normal(0, 0.3)
                    ))

        return path

    for _ in range(density):

        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        edge_strength = edges[y, x] / 255

        prob = (tone ** 2.8) * 1.8 + edge_strength * 1.2
        prob = min(max(prob, 0), 1)

        if np.random.rand() < prob:
            line = trace_line(x, y)
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

    edges = cv2.Canny(gray, 50, 130)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
    angle = np.arctan2(gy, gx)

    points = generate_points(gray, angle, edges, int(density * 3), chaos, [])

    pts = np.array(points)

    if len(pts) < 100:
        raise ValueError("Too few points")

    MAX_POINTS = min(int(density * 12), 22000)

    xs = np.clip(pts[:,0].astype(int), 0, gray.shape[1]-1)
    ys = np.clip(pts[:,1].astype(int), 0, gray.shape[0]-1)

    tones = 1 - gray[ys, xs] / 255
    keep_prob = 0.25 + tones * 0.75

    mask = np.random.rand(len(pts)) < keep_prob
    pts = pts[mask]

    if len(pts) > MAX_POINTS:
        idx = np.random.choice(len(pts), MAX_POINTS, replace=False)
        pts = pts[idx]

    # -----------------------------
    # TSP (SAFE + SAME LOOK)
    # -----------------------------
    used = np.zeros(len(pts), dtype=bool)
    path = []

    current = np.random.randint(len(pts))
    direction = np.array([1.0, 0.0])

    path.append(tuple(pts[current]))
    used[current] = True

    total_steps = len(pts)
    k = 10

    for _ in range(len(pts)-1):

        if len(path) > MAX_POINTS:
            break

        current_point = pts[current]

        dists = np.sum((pts - current_point)**2, axis=1).astype(float)

        # 🔧 stability fix
        dists[dists < 1e-6] = 1e-6
        dists[used] = np.inf

        nearest = np.argpartition(dists, k)[:k]
        nearest = nearest[np.isfinite(dists[nearest])]

        if len(nearest) == 0:
            break

        candidates = pts[nearest]

        local_dists = dists[nearest]

        min_dist = max(np.min(local_dists), 1e-6)
        dist_score = np.exp(-local_dists / min_dist)

        vectors = candidates - current_point
        norms = np.linalg.norm(vectors, axis=1)
        norms[norms < 1e-6] = 1e-6

        unit_vectors = vectors / norms[:, None]
        dir_unit = direction / (np.linalg.norm(direction) + 1e-6)

        alignment = np.dot(unit_vectors, dir_unit)
        dir_score = (alignment + 1) / 2

        xs = np.clip(candidates[:,0].astype(int), 0, gray.shape[1]-1)
        ys = np.clip(candidates[:,1].astype(int), 0, gray.shape[0]-1)

        tones = 1 - gray[ys, xs] / 255
        edge_vals = edges[ys, xs] / 255

        progress = len(path) / total_steps
        edge_phase = max(0, 1 - progress * 2.5)

        scores = (
            dist_score * 0.2 +
            dir_score * 0.5 +
            tones * (0.3 * (1 - edge_phase)) +
            edge_vals * (0.2 + edge_phase * 0.8)
        )

        # 🔧 safety fix (no NaN crash)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

        total = np.sum(scores)

        if total <= 0:
            next_index = np.random.choice(nearest)
        else:
            probs = scores / total
            next_index = np.random.choice(nearest, p=probs)

        new_direction = pts[next_index] - current_point
        direction = direction * 0.75 + new_direction * 0.25

        path.append(tuple(pts[next_index]))
        used[next_index] = True
        current = next_index

    pts = np.array(path)

    # -----------------------------
    # SMOOTH
    # -----------------------------
    try:
        tck, _ = splprep([pts[:,0], pts[:,1]], s=smoothness * 0.05)
        u = np.linspace(0,1,len(pts))
        x, y = splev(u, tck)
        smooth_path = list(zip(x,y))
    except:
        smooth_path = path

    # -----------------------------
    # SCALE
    # -----------------------------
    width_mm, height_mm = PRINT_SIZES.get(size, (210,297))
    if orientation == "landscape":
        width_mm, height_mm = height_mm, width_mm

    pts = np.array(smooth_path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    scale = min(
        (width_mm-20)/(max_x-min_x),
        (height_mm-20)/(max_y-min_y)
    )

    scaled = [
        ((x-min_x)*scale+10, (y-min_y)*scale+10)
        for x,y in pts
    ]

    dwg = svgwrite.Drawing(output_path)
    dwg.viewbox(0,0,width_mm,height_mm)

    dwg.add(dwg.polyline(
        scaled,
        stroke="black",
        fill="none",
        stroke_width=0.2
    ))

    dwg.save()

    save_preview_from_path(
        scaled,
        output_path.replace(".svg",".png"),
        size=size,
        orientation=orientation
    )