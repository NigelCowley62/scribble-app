from flask import Flask, render_template, request, send_file
import os
from processor import process_image_to_svg

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
OUTPUT_FILE = "output.svg"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["image"]

    density = int(request.form.get("density", 1200))
    smooth = float(request.form.get("smooth", 5))
    chaos = float(request.form.get("chaos", 0.2))

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    process_image_to_svg(
        input_path,
        OUTPUT_FILE,
        density=density,
        smoothness=smooth,
        chaos=chaos
    )

    return {"svg_url": "/download"}

@app.route("/download")
def download():
    return send_file(OUTPUT_FILE)

if __name__ == "__main__":
    app.run()