import json
import re
import torch
from tqdm import tqdm
from transformers import pipeline

# Load data
with open("conversation_data_cleaned.json") as f:
    data = json.load(f)

# Enhanced regex topics (from previous artifact)
regex_topics = {
    "delay": [
        "delay", "delayed", "late", "behind schedule", "running late",
        "cancel", "cancelled", "canceled", "cancellation", 
        "reschedule", "rescheduled", "rescheduling",
        "missed connection", "connection", "layover", "wait", "waiting",
        "departure", "arrived late", "departure time", "arrival time",
        "weather delay", "mechanical", "crew", "gate", "boarding"
    ],
    
    "luggage": [
        "luggage", "baggage", "bag", "bags", "suitcase", "carry on", "carryon",
        "lost bag", "missing bag", "lost luggage", "missing luggage",
        "baggage claim", "claim", "lost baggage", "missing baggage",
        "damaged", "broken", "damaged bag", "broken luggage",
        "track", "tracking", "deliver", "delivery", "locate"
    ],
    
    "booking": [
        "booking", "book", "reservation", "reserve", "ticket", "tickets",
        "seat", "seats", "seating", "seat assignment", "upgrade",
        "price", "cost", "fee", "charge", "payment", "refund", "credit",
        "fare", "pricing", "expensive", "cheap",
        "change", "modify", "switch", "transfer", "exchange",
        "name change", "date change", "flight change",
        "check in", "checkin", "online check", "boarding pass", "confirmation"
    ],
    
    "customer_service": [
        "rude", "unprofessional", "terrible", "awful", "horrible", "worst",
        "poor service", "bad service", "disappointed", "frustrated",
        "agent", "staff", "employee", "representative", "rep", "attendant",
        "supervisor", "manager", "crew", "pilot", "gate agent",
        "help", "support", "assistance", "service", "customer service",
        "complaint", "complain", "issue", "problem", "resolve",
        "call", "phone", "hold", "waiting", "response", "reply", "contact"
    ]
}

# Initialize zero-shot classifier (loaded once for efficiency)
print("Loading zero-shot classification model...")
try:
    classifier = pipeline("zero-shot-classification", 
                         model="joeddav/xlm-roberta-large-xnli",
                         device=0 if torch.cuda.is_available() else -1)
    print("✓ Zero-shot model loaded successfully")
except Exception as e:
    print(f"✗ Error loading zero-shot model: {e}")
    print("Falling back to regex-only classification")
    classifier = None

# Zero-shot labels
zero_shot_labels = [
    "flight delay or cancellation",
    "luggage and baggage issues", 
    "booking and reservation problems",
    "customer service complaints"
]

label_mapping = {
    "flight delay or cancellation": "delay",
    "luggage and baggage issues": "luggage", 
    "booking and reservation problems": "booking",
    "customer service complaints": "customer_service"
}

def classify_with_regex(text):
    """Enhanced regex classifier with scoring"""
    text = text.lower()
    topic_scores = {topic: 0 for topic in regex_topics.keys()}
    
    for topic, keywords in regex_topics.items():
        for kw in keywords:
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text))
            weight = len(kw.split()) * 1.5 if len(kw.split()) > 1 else 1
            topic_scores[topic] += matches * weight
    
    best_topic = max(topic_scores, key=topic_scores.get)
    best_score = topic_scores[best_topic]
    
    return best_topic if best_score > 0 else "other", best_score

def batch_zero_shot_classify(texts, threshold=0.4, batch_size=32):
    """Batch zero-shot classification for efficiency"""
    if classifier is None:
        return ["other"] * len(texts)
    
    results = []
    
    try:
        # Process in batches for memory efficiency
        for i in tqdm(range(0, len(texts), batch_size), desc="Zero-shot batches"):
            batch_texts = texts[i:i+batch_size]
            
            # Truncate long texts
            batch_texts = [text[:400] if len(text) > 400 else text for text in batch_texts]
            
            # Batch classification
            batch_results = classifier(batch_texts, zero_shot_labels)
            
            # Process batch results
            for result in batch_results:
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                if top_score < threshold:
                    results.append("other")
                else:
                    mapped_label = label_mapping.get(top_label, top_label)
                    results.append(mapped_label)
        
        return results
    
    except Exception as e:
        print(f"Batch zero-shot classification error: {e}")
        return ["other"] * len(texts)

print("Starting conservative hybrid classification...")

# Step 1: Apply regex to all conversations
print("Step 1: Applying regex classification...")
regex_results = []
texts_for_zero_shot = []
zero_shot_indices = []

for i, convo in enumerate(tqdm(data, desc="Regex classification")):
    text = convo["first_tweet"]
    regex_topic, regex_score = classify_with_regex(text)
    
    if regex_topic != "other":
        regex_results.append((i, regex_topic, "regex"))
    else:
        # Store for batch zero-shot processing
        texts_for_zero_shot.append(text)
        zero_shot_indices.append(i)

print(f"Regex classified: {len(regex_results)} conversations")
print(f"Need zero-shot for: {len(texts_for_zero_shot)} conversations")

# Step 2: Batch process texts that need zero-shot
zero_shot_results = []
if texts_for_zero_shot:
    print("Step 2: Batch zero-shot classification...")
    zero_shot_topics = batch_zero_shot_classify(texts_for_zero_shot)
    zero_shot_results = [(idx, topic, "zero_shot") for idx, topic in zip(zero_shot_indices, zero_shot_topics)]

# Step 3: Combine results and assign to conversations
print("Step 3: Combining results...")
all_results = regex_results + zero_shot_results
all_results.sort(key=lambda x: x[0])  # Sort by original index

method_stats = {"regex": 0, "zero_shot": 0}

for i, (idx, topic, method) in enumerate(all_results):
    data[idx]["topic"] = topic
    data[idx]["classification_method"] = method
    method_stats[method] += 1

# Print results
print(f"\nConservative Hybrid Classification Results:")
print("=" * 50)

# Save results
with open("conversation_data_hybrid_conservative.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nResults saved to conversation_data_hybrid_conservative.json")