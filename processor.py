import cv2
import numpy as np
import svgwrite
from scipy.interpolate import splprep, splev

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
# PREVIEW GENERATION (HIGH RES)
# -----------------------------
def save_preview_from_path(path, png_path, size="A4", orientation="portrait"):

    pts = np.array(path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    data_w = max_x - min_x
    data_h = max_y - min_y

    data_w = max(data_w, 1e-6)
    data_h = max(data_h, 1e-6)

    # preview resolution (high quality)
    if size == "A4":
        base = 1400
    elif size == "A3":
        base = 1800
    elif size == "A2":
        base = 2200
    else:
        base = 2600

    # orientation-aware canvas
    if orientation == "landscape":
        canvas_w = int(base * (data_w / data_h))
        canvas_h = base
    else:
        canvas_w = base
        canvas_h = int(base * (data_h / data_w))

    img = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # scale + center
    scale = min(canvas_w / data_w, canvas_h / data_h)

    offset_x = (canvas_w - data_w * scale) / 2
    offset_y = (canvas_h - data_h * scale) / 2

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
        (50, img.shape[0] // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        3,
        (200, 200, 200),
        4,
        cv2.LINE_AA
    )

    cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
    cv2.imwrite(image_path, img)


# -----------------------------
# FLOW FIELD POINT GENERATION
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

        # 🔥 compute initial tone for stroke length
        ix0, iy0 = int(x), int(y)
        if ix0 < 0 or iy0 < 0 or ix0 >= w or iy0 >= h:
            return path

        tone = 1 - gray[iy0, ix0] / 255

        max_len = int(15 + tone * 60)  # 20–60 depending on tone

        for _ in range(max_len):

            ix, iy = int(x), int(y)

            if ix < 0 or iy < 0 or ix >= w or iy >= h:
                break

            path.append((x, y))

            tone = 1 - gray[iy, ix] / 255
            edge_strength = edges[iy, ix] / 255

            # -----------------------------
            # 🔥 TONAL BAND MAPPING (FIXED)
            # -----------------------------
            if tone < 0.3:
                # highlights (very light)
                weight = tone * 0.3 + edge_strength * 0.4

            elif tone < 0.7:
                # midtones
                weight = tone * 0.7 + edge_strength * 0.8

            else:
                # shadows (boost heavily)
                weight = (tone ** 2.5) * 2.2 + edge_strength * 1.5

            # face boost
            if in_face(x, y):
                weight *= 1.2

            # clamp
            weight = min(max(weight, 0), 1)

            theta = angle[iy, ix] + np.pi / 2

            # less chaos in strong features (faces, edges)

            local_chaos = chaos

            if in_face(x, y):
                local_chaos *= 0.4   # calmer lines in faces

            noise_scale = local_chaos * (0.6 * (1 - weight) + 0.2)
            noise_scale = max(noise_scale, 0.001)  # prevent negative/zero

            theta += np.random.normal(0, noise_scale)

            # smaller steps in dark areas = more detail
            step = 0.1 + (1 - tone) * 0.6
            
            x += np.cos(theta) * step
            y += np.sin(theta) * step

            # 🔥 linger in dark areas (extra density)
            if weight > 0.65:
                x += np.cos(theta) * step * 0.3
                y += np.sin(theta) * step * 0.3

        return path

    seeds = []

    for _ in range(density):
        x = np.random.randint(0, w)
        y = np.random.randint(0, h)

        tone = 1 - gray[y, x] / 255
        edge_strength = edges[y, x] / 255

        # -----------------------------
        # 🔥 TONAL BIASED SEEDING
        # -----------------------------
        if tone < 0.3:
            prob = tone * 0.2 + edge_strength * 0.3

        elif tone < 0.7:
            prob = tone * 0.9 + edge_strength * 0.9

        else:
            prob = (tone ** 2.5) * 1.8 + edge_strength * 1.5

        if in_face(x, y):
            prob *= 1.2

        prob = min(max(prob, 0), 1)

        if np.random.rand() < prob:
            seeds.append((x, y))


    # fallback safety (keep this)
    if len(seeds) < 50:
        for _ in range(200):
            seeds.append((np.random.randint(0, w), np.random.randint(0, h)))


    # trace lines
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
    

    img = cv2.imread(input_path)
    if img is None:
        raise ValueError("Could not load image")

    # preserve aspect ratio
    h, w = img.shape[:2]
    max_dim = 400

    scale = max_dim / max(h, w)

    new_w = int(w * scale)
    new_h = int(h * scale)

    img = cv2.resize(img, (new_w, new_h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -----------------------------
    # 🔥 CONTRAST ENHANCEMENT (CLAHE)
    # -----------------------------
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # -----------------------------
    # 🔥 GAMMA (boost darks)
    # -----------------------------
    gamma = 0.8  # <1 = darker shadows
    gray = np.array(255 * (gray / 255) ** gamma, dtype='uint8')

    # face detection AFTER contrast
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # -----------------------------
    # 🔥 MIDTONE BOOST
    # -----------------------------
    gray = np.clip(gray * 1.1, 0, 255).astype(np.uint8)

    # -----------------------------
    # 🔥 UNSHARP MASK (edge contrast)
    # -----------------------------
    blur_small = cv2.GaussianBlur(gray, (0, 0), 1.0)
    gray = cv2.addWeighted(gray, 1.7, blur_small, -0.7, 0)

    # slight blur for flow field only
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 130)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=5)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5)
    angle = np.arctan2(gy, gx)

    # multi-pass
    # -----------------------------
    # MULTI-LAYER SCRIBBLE (UPGRADED)
    # -----------------------------
    points_light = generate_points(
        gray, angle, edges,
        int(density * 1.2),
        chaos * 1.2,
        faces
    )

    points_mid = generate_points(
        gray, angle, edges,
        int(density * 1.6),
        chaos * 0.9,
        faces
    )

    points_dark = generate_points(
        gray, angle, edges,
        int(density * 2.8),
        chaos * 0.3,
        faces
    )

    # combine layers
    points = points_light + points_mid + points_dark

    # subsample (controls density / prevents overload)

    if len(points) < 50:
        raise ValueError("Not enough points generated")
    pts = np.array(points)
    
    # 🔥 LIMIT POINT COUNT (prevents freezing)
    max_points = 14000

    if len(pts) > max_points:
        tones = 1 - gray[pts[:,1].astype(int), pts[:,0].astype(int)] / 255

        # tone-aware keep probability
        keep_prob = 0.3 + tones * 0.7

        mask = np.random.rand(len(pts)) < keep_prob
        pts = pts[mask]

        # hard cap (guarantee performance)
        if len(pts) > max_points:
            idx = np.random.choice(len(pts), max_points, replace=False)
            pts = pts[idx]

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
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s = smoothness * 0.01)
        u = np.linspace(0, 1, len(pts))
        x, y = splev(u, tck)
        smooth_path = list(zip(x, y))
    except:
        smooth_path = path

    # -----------------------------
    # SCALE TO PRINT SIZE (FIXED)
    # -----------------------------
    width_mm, height_mm = PRINT_SIZES.get(size, (210, 297))

    if orientation == "landscape":
        width_mm, height_mm = height_mm, width_mm

    margin = 10

    draw_w = width_mm - 2 * margin
    draw_h = height_mm - 2 * margin

    print("SIZE:", size)
    print("ORIENTATION:", orientation)
    print("PAGE MM:", width_mm, height_mm)
    print("DRAW AREA:", draw_w, draw_h)
    

    pts = np.array(smooth_path)

    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    data_w = max_x - min_x
    data_h = max_y - min_y

    data_w = max(data_w, 1e-6)
    data_h = max(data_h, 1e-6)

    # preserve aspect ratio
    scale = min(draw_w / data_w, draw_h / data_h)

    offset_x = margin + (draw_w - data_w * scale) / 2
    offset_y = margin + (draw_h - data_h * scale) / 2

    scaled = []
    for x, y in pts:
        px = (x - min_x) * scale + offset_x
        py = (y - min_y) * scale + offset_y
        scaled.append((px, py))

    print("FIRST SCALED POINT:", scaled[0])

    # -----------------------------
    # SAVE SVG
    # -----------------------------
    dwg = svgwrite.Drawing(output_path)

    # 🔥 CRITICAL: define coordinate system properly
    dwg.viewbox(0, 0, width_mm, height_mm)

    # physical size
    dwg['width'] = f"{width_mm}mm"
    dwg['height'] = f"{height_mm}mm"

    dwg.add(dwg.polyline(
        scaled,
        stroke="black",
        fill="none",
        stroke_width=0.2
    ))

    dwg.save()

    # -----------------------------
    # SAVE PREVIEW (MATCH SVG)
    # -----------------------------
    png_path = output_path.replace(".svg", ".png")

    save_preview_from_path(
        scaled,
        png_path,
        size=size,
        orientation=orientation
    )