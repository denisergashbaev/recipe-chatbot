.PHONY: run-app
run-app:
	uvicorn backend.main:app --reload 