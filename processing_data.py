import os
import pandas as pd

def batch_to_df(batch):
   return pd.DataFrame(batch)


def process_and_append(batch, shared_list):
    df = batch_to_df(batch)
    print(f"[PID {os.getpid()}] Processed chunk with {len(df)} rows", flush=True)
    shared_list.append(df)