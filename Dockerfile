FROM python:3.12-slim

WORKDIR /app

# No build-essential/system deps needed: every pinned dependency (including
# bcrypt and psycopg[binary]) ships a manylinux wheel for cp312 -- verified
# with `pip install --only-binary=:all: -r requirements.txt` before removing
# this. Keeps the image smaller and the Railway build faster. If a future
# dependency bump ever needs to compile from source, pip's error message
# will say so explicitly and this step can come back.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN mkdir -p uploads

EXPOSE 8000

# Railway (and most PaaS platforms) inject their own $PORT and route traffic
# to it -- hardcoding 8000 here would make the container unreachable even
# though it's running fine internally. Falls back to 8000 for local
# `docker run` where no PORT is set.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
