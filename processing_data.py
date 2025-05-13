import os
import pandas as pd
from bson import json_util, ObjectId
import json
from pandas import json_normalize

def process_batch(batch):
    sanitised = json.loads(json_util.dumps(batch))
    normalised = json_normalize(sanitised)
    df = pd.DataFrame(normalised)

    return df