import numpy as np
import pandas as pd
import os

import requests
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from src.gap_analysis.skill_gap_analyzer import top_n_resumes_per_jd
from src.skill_extraction.skill_extractor import fetch_feedback
from src.preprocessing.resume_parser import preprocess_resumes
from src.preprocessing.job_loader import preprocess_jobs
from src.preprocessing.text_cleaner import clean_text,pdf_to_text
from src.matching.similarity_calculator import compatibility_score

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    jd_files        = request.files.getlist("jd")
    resume_files    = request.files.getlist("resumes")
    top_n           = request.form.get("top_n", type=int)

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

    scores = list()
    jds = list()
    for path in jd_paths:
        jds.append(pdf_to_text(path))

    resumes = list()
    for path in resume_paths:
        resumes.append(pdf_to_text(path))

    scores = compatibility_score(resumes, jds)

    jd_res_filtered = top_n_resumes_per_jd(jds, resumes, scores, top_n)

    result = []
    jd_names_filtered = []
    for jd_idx, row in enumerate(jd_res_filtered):
        jd_names_filtered.append(os.path.basename(jd_paths[jd_idx]))
        result.append(list())
        for res in row["resumes"]:
            score = int(res["score"])
            # feedback = fetch_feedback(row["jd"], res["resume"], score)
            feedback = "Note: < Disabled due to compuational limit >"
            result[-1].append({"score": score, 
                               "res_name": os.path.basename(resume_paths[res["index"]]), 
                               "label": res["label"],
                               "feedback": feedback})

    app.logger.info(jd_names_filtered)
    return render_template("result.html", jd_names=jd_names_filtered, result=result)

if __name__ == "__main__":
    app.run(debug=True)