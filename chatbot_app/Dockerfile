# 
FROM python:3.10

# 
WORKDIR /code

# 
COPY ./requirements.txt /code/requirements.txt


# 
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# 
COPY ./app /code/app

# Make port 80 available to the world outside this container
EXPOSE 80

# 
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
# CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]