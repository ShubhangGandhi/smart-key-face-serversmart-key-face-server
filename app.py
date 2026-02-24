from fastapi import FastAPI, File, UploadFile, Form
import os
import shutil

app = FastAPI()

# create faces folder
os.makedirs("faces", exist_ok=True)

@app.get("/")
def home():
    return {"status": "running"}


@app.post("/enroll")
async def enroll(name: str = Form(...), file: UploadFile = File(...)):

    folder = f"faces/{name}"

    os.makedirs(folder, exist_ok=True)

    file_path = f"{folder}/{file.filename}"

    with open(file_path, "wb") as buffer:

        shutil.copyfileobj(file.file, buffer)

    return {"status": "success", "message": "Face enrolled"}


@app.post("/recognize")
async def recognize(file: UploadFile = File(...)):

    # temporary response

    return {"result": "unknown"}
