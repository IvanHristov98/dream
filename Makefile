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
	@pylint dream --max-line-length=120 \
		--disable=too-many-locals \
		--disable=missing-function-docstring \
		--disable=missing-module-docstring \
		--disable=missing-class-docstring \
		--disable=too-few-public-methods \
		--disable=fixme \
		--disable=no-member

.PHONY: fmt
fmt:
	@black ./dream -l 120
	@black ./cmd -l 120
	@black ./tests -l 120

clean-imstore:
	$(call clean_dir_except_file,${PWD}/tmp/imstore,.gitignore)

define clean_dir_except_file
	@echo "Cleaning $(1) except $(2)..."
	@find "$(1)" -mindepth 1 | grep -v "$(2)" | xargs -I {} rm {}
endef

test:
	@cd tests/voctree && PYTHONPATH="${PYTHONPATH}:${PWD}" behave --format="progress" && cd -

test-wip:
	@cd tests/voctree && PYTHONPATH="${PYTHONPATH}:${PWD}" behave -w && cd -

run-server:
	@VTREE_TRAIN_PROC_COUNT=1 \
		IM_STORE_PATH="$(PWD)/tmp/imstore" \
		python3 cmd/server.py

seed:
	@python3 cmd/seed.py  \
		-coco2014-captions-path="$(PWD)/data/coco2014/captions_train2014.json" \
		-coco2014-ims-path="$(PWD)/data/coco2014/train2014" \
		-imstore-ims-path="$(PWD)/tmp/imstore"
