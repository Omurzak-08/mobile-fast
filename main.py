import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import joblib



model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler (8).pkl')

app = FastAPI()


class PhoneSchema(BaseModel):
    Rating: float
    Num_Ratings: int
    RAM: float
    ROM: float
    Back_Cam: int
    Front_Cam: int
    Battery: float


@app.post("/predict")
def predict(phone: PhoneSchema):
    FEATURES = ['Rating', 'RAM', 'ROM', 'Num_Ratings', 'Front_Cam', 'Back_Cam', 'Battery']


    phone_dict = phone.model_dump()
    features = [phone_dict[feature] for feature in FEATURES]


    prediction = model.predict([features])

    return {"predicted_price_inr": prediction.tolist()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
