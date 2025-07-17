import os
from flask import Flask, render_template, request, send_from_directory
from infer import infer_one_image 

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

MODEL_CKPT = 'checkpoints/unet_best.pth'

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400
        file = request.files["file"]
        if file.filename == "":
            return "No file selected", 400
        input_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(input_path)
        output_filename = "result_" + file.filename
        output_path = os.path.join(RESULT_FOLDER, output_filename)
        infer_one_image(input_path, MODEL_CKPT, output_path)
        return render_template("index.html", result_img=output_filename)
    return render_template("index.html")

@app.route("/results/<filename>")
def result_file(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
