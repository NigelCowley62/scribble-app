from flask import Flask, render_template, request, send_file, jsonify
import os
from processor import process_image_to_svg, add_watermark

app = Flask(__name__)

# -----------------------------
# PATHS
# -----------------------------
UPLOAD_FOLDER = "uploads"
OUTPUT_SVG = "output.svg"
OUTPUT_PNG = "output.png"  # must match processor output

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# -----------------------------
# HOME
# -----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -----------------------------
# UPLOAD + PROCESS
# -----------------------------
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    # sliders
    density = int(request.form.get("density", 1200))
    smooth = float(request.form.get("smooth", 5))
    chaos = float(request.form.get("chaos", 0.2))

    # optional UI inputs (not used yet)
    size = request.form.get("size", "A4")
    orientation = request.form.get("orientation", "portrait")
    print("FORM ORIENTATION:", orientation)

    # save uploaded image
    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # run processor (creates SVG + PNG)
    process_image_to_svg(
        input_path,
        OUTPUT_SVG,
        density=density,
        smoothness=smooth,
        chaos=chaos,
        size=size,
        orientation=orientation
    )

    # add watermark to preview
    add_watermark(OUTPUT_PNG)

    # return preview
    return jsonify({"preview_url": "/preview"})


# -----------------------------
# SERVE PREVIEW
# -----------------------------
@app.route("/preview")
def preview():
    return send_file(OUTPUT_PNG, mimetype="image/png")


# -----------------------------
# OPTIONAL DOWNLOAD (for testing)
# -----------------------------
@app.route("/download")
def download():
    return send_file(OUTPUT_SVG, as_attachment=True)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)