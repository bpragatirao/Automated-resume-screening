import numpy as np
import pandas as pd
import os

import requests
from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.utils import secure_filename

from src.skill_extraction.skill_extractor import fetch_feedback
from src.preprocessing.resume_parser import preprocess_resumes
from src.preprocessing.job_loader import preprocess_jobs
from src.preprocessing.text_cleaner import clean_text,extract_text
from src.matching.similarity_calculator import compatibility_score

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

TOP_N       = 1
JD_LIST     = []
RES_LIST    = []
SIM_SCORES  = None
SORTED_LIST = None

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    global TOP_N, JD_LIST, RES_LIST, SORTED_LIST, SIM_SCORES

    JD_LIST.clear()
    RES_LIST.clear()

    jd_files      = request.files.getlist('jd_files')
    resume_files  = request.files.getlist('resume_files')
    TOP_N         = request.form.get("resume_count", 1, type=int)
    jd_texts      = request.form.getlist('jd_texts')
    resume_texts  = request.form.getlist('resume_texts')

    jd_texts_header = []
    resume_texts_header = [f"resume_text_{i}" for i in range(len(resume_texts))]

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

    for i, t in enumerate(jd_texts):
        JD_LIST.append((f"job_description_text_{i}", t))
    for p in jd_paths:
        JD_LIST.append((os.path.basename(p), extract_text(p)))

    for i, t in enumerate(resume_texts):
        RES_LIST.append((f"job_description_text_{i}", t))
    for p in resume_paths:
        RES_LIST.append((os.path.basename(p), extract_text(p)))

    SIM_SCORES = compatibility_score(
            [item[1] for item in RES_LIST],
            [item[1] for item in JD_LIST]
            )

    # Note(anb): 2D list with res index in desc order of score per jd
    #           0: [res1, res2, ..] -> res in desc order for jd0

    score = np.asarray(SIM_SCORES)
    SORTED_LIST = [np.argsort(score[i])[::-1].tolist() for i in range(score.shape[0])]
    return redirect(url_for('results'))

@app.route('/results')
def results():
    jd_names = [item[0] for item in JD_LIST]
    result = []
    for i in range(len(jd_names)):
        vals = []
        for cnt, idx in enumerate(SORTED_LIST[i]):
            vals.append({
                "res_name": RES_LIST[idx][0],
                "score": SIM_SCORES[i][idx],
                "label": "pass" if cnt < TOP_N else "fail"
                })
        result.append(vals)

    return render_template("result.html", 
                           jd_names=jd_names, 
                           result=result)

@app.route('/get_feedback', methods=['POST'])
def get_feedback():
    data = request.get_json()
    #Note(anb): jth jd ith resume
    jd_idx = int(data.get('jd_index'))
    res_idx_rel = int(data.get('resume_index'))

    res_idx = SORTED_LIST[jd_idx][res_idx_rel]
    feedback = fetch_feedback(JD_LIST[jd_idx][1], RES_LIST[res_idx][1], SIM_SCORES[jd_idx][res_idx])
    # feedback = "No Response: disabled generative-model for now"

    return jsonify({
        "success": True,
        "feedback": feedback
    })

if __name__ == "__main__":
    app.run(debug=True)
