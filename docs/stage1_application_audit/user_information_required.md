# Outstanding Information Required from User — Stage 1 Application Audit

**Project:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System  
**Document Location:** `docs/stage1_application_audit/user_information_required.md`  
**Related Workbook:** [`docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx`](file:///Users/rahulmanchanda/Desktop/AI-Driven-Satellite-Image-Semantic-Search-and-Temporal-Change-Detection-System/docs/stage1_application_audit/Stage1_Application_Audit_Completed.xlsx)  
**Date:** November 13, 2026  

---

## Instructions for the User

The technical and experimental forensic audit of your project repository is **100% complete**. All source code, model configurations, checkpoints, experimental result artifacts (JSON/CSV), automated tests, and research figures have been verified and documented.

In accordance with strict audit rules, **no personal, institutional, or administrative metadata was fabricated or guessed**.

Please fill in the 5 administrative fields below to finalize your Stage 1 Application submission:

---

## Required Administrative Information

| Field # | Information Item | Status in Codebase | User Input Needed |
|---|---|---|---|
| **1** | **Primary Student / Project Lead** | `VERIFIED` | Rahul Manchanda *(Please confirm full formal name if different)* |
| **2** | **Project Team Collaborators** | `VERIFIED` | 1. Tanishka Mukhi (Dataset & Architecture Lead)<br>2. Adishri Abro (Literature & Evaluation Lead)<br>*(Please confirm formal names and student IDs/roll numbers)* |
| **3** | **Academic Institution / College** | `REQUIRES USER INPUT` | *[Please enter your college/institution name, e.g., Department of Computer Science, ABC Institute of Technology]* |
| **4** | **Affiliated University** | `REQUIRES USER INPUT` | *[Please enter affiliated university name, e.g., XYZ University]* |
| **5** | **Academic Year & Semester** | `REQUIRES USER INPUT` | *[Please enter academic year, e.g., 2026–2027, Final Year B.Tech / M.Tech]* |
| **6** | **Project Supervisor / Faculty Guide** | `REQUIRES USER INPUT` | *[Please enter Guide Name, Designation, and Department]* |
| **7** | **Production Cloud URL (Optional)** | `REQUIRES USER INPUT` | *[If the Vercel application is deployed to an active public URL, provide it here; otherwise mark as 'Local Research Deployment Only']* |

---

## Summary of Fully Verified Technical Fields (No Input Needed)

To assist you during submission, the following technical fields have been verified from repository evidence:
- **Project Title:** AI-Driven Satellite Image Semantic Search and Temporal Change Detection System
- **Core Datasets:** RSICD (1,093 test scenes, 5,465 human captions) & LEVIR-CD (128 test scenes of $1024 	imes 1024$ px, $134,217,728$ pixels total).
- **Retrieval Architecture:** CLIP ViT-B/32, 512-D L2-normalized embeddings, FAISS IndexFlatIP exact cosine search, 5 remote sensing prompt templates.
- **Change Detection Models:** 7 evaluated models: B1 (Pixel Diff), B2 (SSIM), B3 (CVA), M7 (FC-Siam-diff), M8 (SNUNet-CD), M9 (BIT-CD), M10 (QAT-CD).
- **Quality-Aware Gating (QAT-CD):** $P_{	ext{QAT}} = P_0 \cdot (1 - \gamma \cdot U \cdot Q_{	ext{defect}})$, achieving $-29.19\%$ false-positive pixel reduction at $\gamma=0.30$.
- **Temporal Understanding Pipeline:** Two-stage ORB + ECC co-registration ($	au_{	ext{conf}} \ge 0.40$), morphological 8-connected component clustering, GSD-based physical area ($m^2$), Visible Atmospherically Resistant Index (VARI), 8-class taxonomy, and dual-stage Groq Qwen 27B explanation synthesis.
- **Hardware Profile:** Apple Silicon arm64 8-core CPU, 4 PyTorch intra-op threads, $<8	ext{ GB}$ RAM constraint (Peak RSS $877.3	ext{ MB}$).
