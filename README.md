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

python3 cmd/seed.py  -coco2014-captions-path="${PWD}/data/coco2014/captions_train2014.json" -coco2014-ims-path="${PWD}/data/coco2014/train2014" -imstore-ims-path="${PWD}/tmp/imstore"
```
