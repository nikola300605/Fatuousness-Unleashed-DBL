import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pymongo_interface import update_one_document
from bson import ObjectId
from tqdm import tqdm

with open("conversation_data_resolved.json") as f:
    data = json.load(f)

updated = 0
for convo in tqdm(data, desc="Saving to mongoDB"):
    cid = ObjectId(convo.get('conversation_id', None))
    topic = convo.get('topic', None)
    resolved = convo.get('resolved')
    update_data = {
        '_id' : cid,
        'topic' : topic,
        'resolved' : resolved
    }
    doc = update_one_document('conversations', update_data, 'set')
    updated+=doc

print(updated)

