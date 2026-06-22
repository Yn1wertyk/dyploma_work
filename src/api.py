from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.inference import get_detector
import uvicorn

app = FastAPI(title="Fraud Detection API")


class Transaction(BaseModel):
    user_id: str
    amount: float = Field(..., ge=0)
    transaction_type: str
    merchant_category: str
    country: str
    hour: int = Field(..., ge=0, le=23)
    device_risk_score: float = Field(..., ge=0, le=1)
    ip_risk_score: float = Field(..., ge=0, le=1)


class FraudResponse(BaseModel):
    fraud_probability: float
    decision: str
    risk_level: str
    top_features: dict[str, float]
    explanation: str


def detector():
    try:
        return get_detector()
    except FileNotFoundError:
        raise HTTPException(503, "Model is not detected.")
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/")
async def root():
    return {"message": "Fraud Detection API"}


@app.get("/health")
async def health():
    detector()
    return {"status": "healthy"}


@app.post("/score", response_model=FraudResponse)
async def score(tx: Transaction):
    return detector().predict_single(tx.model_dump())


@app.post("/batch_score")
async def batch(transactions: list[Transaction]):
    if not transactions:
        raise HTTPException(400, "List is empty")

    results = detector().predict_batch([t.model_dump() for t in transactions])
    return {"results": results, "count": len(results)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
