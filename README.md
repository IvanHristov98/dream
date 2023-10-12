# dream
Trying to find motivation to program :smile:.

```bash
# For psycopg2
sudo apt install libpq-dev

# To build migrate container
docker build -f Dockerfile.migrate -t yoyo-migrate .
```

To seed the db with a dataset:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export DB_CONN_INFO="host=localhost port=5432 dbname=dream user=dream password=devpass"

```
