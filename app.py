from fastapi import FastAPI, File, UploadFile, Form
import os
import shutil
import uuid
import cv2
import numpy as np
import insightface

app = FastAPI()

FACE_DIR = "faces"

os.makedirs(FACE_DIR, exist_ok=True)

model = insightface.app.FaceAnalysis()
model.prepare(ctx_id=0)

@app.get("/")
def home():
    return {"status": "running"}

@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):

    person_dir = os.path.join(FACE_DIR, name)

    os.makedirs(person_dir, exist_ok=True)

    filename = str(uuid.uuid4()) + ".jpg"

    path = os.path.join(person_dir, filename)

    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"status": "saved"}

@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    contents = await file.read()

    npimg = np.frombuffer(contents, np.uint8)

    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    faces = model.get(img)

    if len(faces) == 0:
        return {"result": "No Face"}

    emb = faces[0].embedding

    best_match = "Unknown"

    best_score = 0

    for person in os.listdir(FACE_DIR):

        person_dir = os.path.join(FACE_DIR, person)

        for image in os.listdir(person_dir):

            path = os.path.join(person_dir, image)

            img_db = cv2.imread(path)

            faces_db = model.get(img_db)

            if len(faces_db) == 0:
                continue

            emb_db = faces_db[0].embedding

            score = np.dot(emb, emb_db)

            if score > best_score:
                best_score = score
                best_match = person

    if best_score > 0.5:
        return {"result": best_match}

    return {"result": "Unknown"}
