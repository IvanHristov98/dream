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
	@black ./dream -l 120
	@black ./cmd -l 120

clean-imstore:
	$(call clean_dir_except_file,${PWD}/tmp/imstore,.gitignore)

define clean_dir_except_file
	@echo "Cleaning $(1) except $(2)..."
	@find "$(1)" -mindepth 1 | grep -v "$(2)" | xargs -I {} rm {}
endef
