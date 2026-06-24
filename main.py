from fastapi import FastAPI
import joblib
import uvicorn
from pydantic import BaseModel

avocado_app = FastAPI()

model = joblib.load('model_avocado.pkl')
scaler = joblib.load('scaler_avocado.pkl')

class AvocadoFeatures(BaseModel):
    firmness: float
    hue: int
    saturation: int
    brightness: int
    sound_db: int
    weight_g: int
    size_cm3: int


@avocado_app.post('/predict')
async def predict_avocado(avocado: AvocadoFeatures):

    features = [
        avocado.firmness,
        avocado.hue,
        avocado.saturation,
        avocado.brightness,
        avocado.sound_db,
        avocado.weight_g,
        avocado.size_cm3,
    ]

    scaled = scaler.transform([features])
    prediction = model.predict(scaled)[0]
    proba = model.predict_proba(scaled)[0]
    classes = model.classes_

    probabilities = {}
    for i in range(len(classes)):
        probabilities[classes[i]] = round(float(proba[i]), 2)

    return {
        'predicted_ripeness': prediction,
        'probabilities': probabilities
    }

if __name__ == '__main__':
    uvicorn.run(avocado_app, host='127.0.0.1', port=9000)