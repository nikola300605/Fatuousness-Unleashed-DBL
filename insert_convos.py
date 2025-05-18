from processing_data import mine_conversations
from pymongo_interface import save_conversations_to_mongo
from tqdm import tqdm


for conv_batch in tqdm(mine_conversations(), desc="Processing conversation batches"):
    save_conversations_to_mongo(conv_batch, 'conversations')