.PHONY: compose-up compose-down smoke

compose-up:
	docker compose up -d --build

compose-down:
	docker compose down --volumes --remove-orphans

smoke: compose-up
	python -m pip install --upgrade pip
	python -m pip install httpx
	python scripts/compose_smoke_test.py
	$(MAKE) compose-down
