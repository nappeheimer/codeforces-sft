from wandb.sdk.internal import datastore
from wandb.proto import wandb_internal_pb2
import os

path = "wandb/latest-run/" + next(
    f for f in os.listdir("wandb/latest-run")
    if f.endswith(".wandb")
)

ds = datastore.DataStore()
ds.open_for_scan(path)

while True:
    data = ds.scan_data()

    if data is None:
        break

    rec = wandb_internal_pb2.Record()
    rec.ParseFromString(data)

    if rec.HasField("history"):
        print(rec.history)
        break
