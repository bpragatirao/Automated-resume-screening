# # app.py
# from flask import Flask, render_template, request
# import json
# from werkzeug.utils import secure_filename

# from sentence_transformers import SentenceTransformer, util
# from demo import compatibility_score, fetch_feedback, pdf_to_text
# import logging

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "uploads"

# MODEL = SentenceTransformer("TechWolf/JobBERT-v2", local_files_only=True)

# import os
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# @app.route("/")
# def index():
#     return render_template("index.html")


# @app.route("/upload", methods=["POST"])
# def upload():
#     jd_files = request.files.getlist("jd")
#     resume_files = request.files.getlist("resumes")

#     jd_paths = []
#     for f in jd_files:
#         path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
#         f.save(path)
#         jd_paths.append(path)

#     resume_paths = []
#     for f in resume_files:
#         path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
#         f.save(path)
#         resume_paths.append(path)

#     reasoning = list()
#     scores = list()

#     jds = list()
#     for path in jd_paths:
#         jds.append(pdf_to_text(path))

#     resumes = list()
#     for path in resume_paths:
#         resumes.append(pdf_to_text(path))

#     scores = compatibility_score(MODEL, resumes, jds)

#     for i in range(len(jds)):
#         reasoning.append([])
#         for j in range(len(resumes)):
#             reasoning[-1].append(["<fetch_feedback func call>"])
#             reasoning[-1].append([fetch_feedback(jds[i], resumes[j], scores[i][j])])

#     return render_template("result.html", scores=scores, reasoning=reasoning)


# if __name__ == "__main__":
#     app.run(debug=True)


# import sys
# import os
# GLOBAL_RESULTS = None
# # current file ka path
# current_dir = os.path.dirname(os.path.abspath(__file__))

# # project root (one level up)
# parent_dir = os.path.abspath(os.path.join(current_dir, ".."))

# # add to python path
# sys.path.insert(0, parent_dir)

# from flask import Flask, render_template, request
# from werkzeug.utils import secure_filename
# import os

# from sentence_transformers import SentenceTransformer

# # import your logic
# from test2 import process_data
# from demo import pdf_to_text   # same function you already have

# app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = "uploads"

# # create upload folder if not exists
# os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# # Load model once (IMPORTANT for performance)
# MODEL = SentenceTransformer("TechWolf/JobBERT-v2", local_files_only=True)


# # ================= HOME =================
# @app.route("/")
# def index():
#     return render_template("index.html")


# # ================= UPLOAD =================
# @app.route("/upload", methods=["POST"])
# def upload():

#     jd_files = request.files.getlist("jd")
#     resume_files = request.files.getlist("resumes")

#     # ===== Save files =====
#     jd_paths = []
#     for f in jd_files:
#         path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
#         f.save(path)
#         jd_paths.append(path)

#     resume_paths = []
#     for f in resume_files:
#         path = os.path.join(app.config["UPLOAD_FOLDER"], secure_filename(f.filename))
#         f.save(path)
#         resume_paths.append(path)

#     # ===== Convert PDF → text =====
#     jds = [pdf_to_text(p) for p in jd_paths]
#     resumes = [pdf_to_text(p) for p in resume_paths]

#     # ===== Run ML pipeline =====
#     results = process_data(MODEL, resumes, jds, top_k=10)
#     global GLOBAL_RESULTS
#     GLOBAL_RESULTS = results
#     # Convert dict → list (easy for frontend)
#     jd_list = list(results.values())

#     return render_template(
#         "result.html",
#         results=jd_list,
#         selected_jd=0
#     )


# # ================= OPTIONAL: JD TAB SWITCH =================
# @app.route("/results/<int:jd_index>")
# def show_results(jd_index):
#     global results  # use same stored results

#     return render_template(
#         "result.html",
#         results=results,
#         selected_jd=jd_index
#     )

# # ================= RUN =================
# if __name__ == "__main__":
#     app.run(debug=True)

