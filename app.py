import sys
import os
GLOBAL_RESULTS = None
# current file ka path
current_dir = os.path.dirname(os.path.abspath(__file__))

# project root (one level up)
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# add to python path
sys.path.insert(0, parent_dir)

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os

from sentence_transformers import SentenceTransformer

# import your logic
from test2 import process_data
from demo import pdf_to_text   # same function you already have



from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import os

from sentence_transformers import SentenceTransformer
from test2 import process_data   # make sure path is correct

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ✅ GLOBAL VARIABLE
GLOBAL_RESULTS = None

# ✅ LOAD MODEL ONCE
MODEL = SentenceTransformer("TechWolf/JobBERT-v2", local_files_only=True)


# ================= HOME =================
@app.route("/")
def index():
    return render_template("index.html")


# ================= UPLOAD =================
@app.route("/upload", methods=["POST"])
def upload():
    global GLOBAL_RESULTS

    jd_files = request.files.getlist("jd")
    resume_files = request.files.getlist("resumes")

    jd_paths = []
    for f in jd_files:
        path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
        f.save(path)
        jd_paths.append(path)

    resume_paths = []
    for f in resume_files:
        path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
        f.save(path)
        resume_paths.append(path)

    # ✅ PROCESS DATA
    GLOBAL_RESULTS = process_data(MODEL, resume_paths, jd_paths)
    # ✅ REDIRECT TO FIRST JD
    return redirect("/results/0")


# ================= RESULTS =================
@app.route("/results/<int:jd_id>")
def show_results(jd_id):
    global GLOBAL_RESULTS

    if GLOBAL_RESULTS is None:
        return "No results found. Please upload files first."

    return render_template(
        "result.html",
        results=GLOBAL_RESULTS,   # ✅ THIS IS IMPORTANT
        selected_jd=jd_id
    )


# ================= RUN =================
if __name__ == "__main__":
    app.run(debug=True)