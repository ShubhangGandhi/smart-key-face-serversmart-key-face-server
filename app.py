from fastapi import FastAPI, File, UploadFile
import os
import cv2
import numpy as np
from deepface import DeepFace

app = FastAPI()

DB="faces"

@app.get("/")
def root():
    return {"message":"Smart Key Server Running"}

@app.post("/recognize")

async def recognize(file:UploadFile=File(...)):

    contents=await file.read()

    nparr=np.frombuffer(contents,np.uint8)

    img=cv2.imdecode(nparr,cv2.IMREAD_COLOR)

    try:

        result=DeepFace.find(img,DB,enforce_detection=False)

        if len(result[0])>0:

            name=result[0]['identity'][0]

            return {"name":name}

        else:

            return {"name":"unknown"}

    except:

        return {"name":"unknown"}
