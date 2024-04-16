# dream

A semantic image searcher.

```bash
# For psycopg2
sudo apt install libpq-dev

# To build migrate container
docker build -f Dockerfile.migrate -t yoyo-migrate .
```

Some commands:

```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}"
export DB_CONN_INFO="host=localhost port=5432 dbname=dream user=dream password=devpass"

make seed # currently using only coco2014
make run-server
```
