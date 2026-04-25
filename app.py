from flask import Flask, render_template, request, send_file, jsonify
import os
import uuid
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

    density = int(request.form.get("density", 1200))
    smooth = float(request.form.get("smooth", 5))
    chaos = float(request.form.get("chaos", 0.2))

    size = request.form.get("size", "A4")
    orientation = request.form.get("orientation", "portrait").lower()

    # generate unique job id
    job_id = str(uuid.uuid4())

    input_path = f"uploads/{job_id}.png"
    output_svg = f"outputs/{job_id}.svg"
    output_png = f"outputs/{job_id}.png"

    file.save(input_path)

    process_image_to_svg(
        input_path,
        output_svg,
        density=density,
        smoothness=smooth,
        chaos=chaos,
        size=size,
        orientation=orientation
    )

    add_watermark(output_png)

    return jsonify({
        "preview_url": f"/preview/{job_id}",
        "download_url": f"/download/{job_id}"
    })


# -----------------------------
# SERVE PREVIEW
# -----------------------------
@app.route("/preview/<job_id>")
def preview(job_id):
    return send_file(f"outputs/{job_id}.png", mimetype="image/png")


# -----------------------------
# OPTIONAL DOWNLOAD (for testing)
# -----------------------------
@app.route("/download/<job_id>")
def download(job_id):
    return send_file(f"outputs/{job_id}.svg", as_attachment=True)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)