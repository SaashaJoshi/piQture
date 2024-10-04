FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN python -m venv venv
RUN . ./venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt
COPY . /app
CMD ["./venv/bin/python", "setup.py" ,"--help-commands"] 