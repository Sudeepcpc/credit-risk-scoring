.PHONY: install train test serve docker
export PYTHONPATH := src:.
install: ; pip install -r requirements.txt
train:   ; python -m credit_risk.train
test:    ; pytest -q
serve:   ; uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
docker:  ; docker build -t credit-risk-api . && docker run -p 8000:8000 credit-risk-api
