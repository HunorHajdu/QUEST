APP_NAME = main.py
VENV_DIR = venv
REQ_FILE = requirements.txt

.PHONY: all setup run clean

all: setup run

setup:
	@echo "Setting up the virtual environment and installing dependencies..."
	python3 -m venv $(VENV_DIR)
	$(VENV_DIR)/bin/pip install -r $(REQ_FILE)

run:
	@echo "Running the Streamlit app..."
	$(VENV_DIR)/bin/streamlit run $(APP_NAME)

clean:
	@echo "Cleaning up the virtual environment..."
	rm -rf $(VENV_DIR)