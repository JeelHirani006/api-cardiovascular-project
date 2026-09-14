from fastapi import FastAPI
import joblib
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://frontend-cardiovascular-ml-project.vercel.app/"], 
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "ML prediction API is running"}


@app.post("/predict")
def predict(data: dict):

    gender = 2 if data["gender"] == "male" else 1

    cholesterol = (
        1 if data["cholesterol"] == "normal"
        else 2 if data["cholesterol"] == "above normal"
        else 3
    )

    gluc = (
        1 if data["gluc"] == "normal"
        else 2 if data["gluc"] == "above normal"
        else 3
    )

    smoke = 1 if data["smoke"] else 0
    alco = 1 if data["alco"] else 0
    active = 1 if data["active"] else 0

    
    features = [
        data["age"],
        gender,
        data["height"],
        data["weight"],
        data["ap_hi"],
        data["ap_lo"],
        cholesterol,
        gluc,
        smoke,
        alco,
        active,
    ]

    prediction = model.predict([features])

    return {
        "prediction": int(prediction[0])
    }