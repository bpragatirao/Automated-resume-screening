# Automated-resume-screening
An NLP-powered recruitment automation system that utilizes Transformer-based models (BERT/S-BERT) for semantic resume screening, skill extraction, and automated skill gap analysis.

# Automated Resume Screening and Skill Gap Analysis Using NLP

## 📌 Project Overview
The recruitment process often involves manual screening of a large number of resumes, making it time-consuming and prone to human bias. This project addresses these limitations by leveraging **Natural Language Processing (NLP)** and deep learning techniques to automate resume screening and identify skill gaps. By using transformer-based models, the system moves beyond traditional keyword matching to capture true semantic relevance and contextual skill representation.

## 👥 Team Members (Group 1)
* **Amrutanshu Bhat**
* **Anurag Kashyap**
* **Ashwin Dhangar**
* **Bammidi Pragati Rao**

**Date:** January 22, 2026  
**Course:** Advanced Machine Learning Mini Project

---

## 📖 Table of Contents
1. [Introduction](#-introduction)
2. [State-of-the-Art Models](#-state-of-the-art-models)
3. [Research Gaps and Motivation](#-research-gaps-and-motivation)
4. [Dataset Analysis](#-dataset-analysis)
5. [Methodology and Approach](#-methodology-and-approach)
6. [Experimental Results and Analysis](#-experimental-results-and-analysis)
7. [Conclusion](#-conclusion)

---

## 🚀 Introduction
### Problem Statement
Manual resume screening is inefficient, subjective, and not scalable. Traditional methods rely on keyword matching, which fails to understand the context of a candidate's experience. There is a critical need for an intelligent system that automatically matches resumes with job descriptions and identifies missing skills.

### Objectives
* Automate resume screening using advanced NLP techniques.
* Extract and normalize skills from unstructured resume data.
* Compute semantic similarity between resumes and job descriptions.
* Perform detailed skill gap analysis to provide actionable feedback.

### Scope
The current system is optimized for **IT and software engineering** job roles but is designed with a modular architecture that can be extended to other domains.

---

## 🧠 State-of-the-Art Models
This project utilizes transformer-based architectures, specifically:
* **BERT (Bidirectional Encoder Representations from Transformers):** Used for understanding the deep context of language.
* **Sentence-BERT (S-BERT):** Employed to generate semantic embeddings and compute relevance scores between resumes and job descriptions efficiently.



---

## 🔬 Methodology and Approach
The system architecture follows a structured NLP pipeline:
1.  **Data Acquisition:** Gathering resume and job description datasets.
2.  **Preprocessing:** Text cleaning, tokenization, and structure normalization.
3.  **Skill Extraction:** Identifying technical competencies and normalizing them for comparison.
4.  **Embedding Generation:** Transforming text into high-dimensional vectors.
5.  **Similarity Computation:** Using cosine similarity to rank resumes against job requirements.
6.  **Skill Gap Analysis:** Highlighting specific requirements present in the JD but missing in the resume.



---

## 📊 Dataset & Results
* **Dataset Analysis:** The project involves rigorous preprocessing of resume datasets to extract statistical insights and ensure model accuracy.
* **Experimental Results:** Comparisons show that this transformer-based approach significantly outperforms traditional keyword-based methods in both accuracy and efficiency.
* **Feedback System:** The system provides actionable feedback by explicitly listing missing skills for candidates.

---

## 🏁 Conclusion
This project successfully demonstrates the power of BERT and S-BERT in modernizing the recruitment process. By automating semantic screening, we reduce human bias and improve the speed of talent acquisition. Future research directions include expanding the domain expertise of the model and integrating real-time recruitment platform APIs.

---
*Developed at NITK Surathkal - Advanced Machine Learning*
