.PHONY: setup
setup: create-venv install-deps

.PHONY: create-venv
create-venv:
	$(call create_venv)

define create_venv
	@if [ ! -d ".venv" ]; then\
		python3 -m venv .venv;\
	fi
endef

.PHONY: install-deps
install-deps:
	@pip install -r requirements.txt

.PHONY: lint
lint:
	@echo "\033[0;33mLinting dream package...\033[0m"
	@pylint dream --max-line-length=120 --disable=too-many-locals

.PHONY: fmt
fmt:
	@black . -l 120
