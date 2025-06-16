import json
import re
from tqdm import tqdm  # tqdm for progress bar

with open("conversation_data_cleaned.json") as f:
    data = json.load(f)

topics = {
    "delay": [
        # Direct delay terms
        "delay", "delayed", "late", "behind schedule", "running late",
        # Cancellation terms
        "cancel", "cancelled", "canceled", "cancellation", 
        "reschedule", "rescheduled", "rescheduling",
        # Time-related issues
        "missed connection", "connection", "layover", "wait", "waiting",
        "departure", "arrived late", "departure time", "arrival time",
        # Weather/operational delays
        "weather delay", "mechanical", "crew", "gate", "boarding"
    ],
    
    "luggage": [
        # Basic luggage terms
        "luggage", "baggage", "bag", "bags", "suitcase", "carry on", "carryon",
        # Lost/missing luggage
        "lost bag", "missing bag", "lost luggage", "missing luggage",
        "baggage claim", "claim", "lost baggage", "missing baggage",
        # Damage issues
        "damaged", "broken", "damaged bag", "broken luggage",
        # Tracking/delivery
        "track", "tracking", "deliver", "delivery", "locate"
    ],
    
    "booking": [
        # Reservation terms
        "booking", "book", "reservation", "reserve", "ticket", "tickets",
        "seat", "seats", "seating", "seat assignment", "upgrade",
        # Payment/pricing
        "price", "cost", "fee", "charge", "payment", "refund", "credit",
        "fare", "pricing", "expensive", "cheap",
        # Changes/modifications
        "change", "modify", "switch", "transfer", "exchange",
        "name change", "date change", "flight change",
        # Check-in related
        "check in", "checkin", "online check", "boarding pass", "confirmation"
    ],
    
    "customer_service": [
        # Service quality
        "rude", "unprofessional", "terrible", "awful", "horrible", "worst",
        "poor service", "bad service", "disappointed", "frustrated",
        # Staff interactions
        "agent", "staff", "employee", "representative", "rep", "attendant",
        "supervisor", "manager", "crew", "pilot", "gate agent",
        # Help/support
        "help", "support", "assistance", "service", "customer service",
        "complaint", "complain", "issue", "problem", "resolve",
        # Communication issues
        "call", "phone", "hold", "waiting", "response", "reply", "contact"
    ]
}

def classify_topic(text):
    """
    Improved classification with better pattern matching and scoring
    """
    text = text.lower()
    
    # Score each topic based on keyword matches
    topic_scores = {topic: 0 for topic in topics.keys()}
    
    for topic, keywords in topics.items():
        for kw in keywords:
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
            matches = len(re.findall(pattern, text))
            
            # Weight longer, more specific keywords higher
            weight = len(kw.split()) * 1.5 if len(kw.split()) > 1 else 1
            topic_scores[topic] += matches * weight
    
    # Return topic with highest score, or "other" if no matches
    best_topic = max(topic_scores, key=topic_scores.get)
    return best_topic if topic_scores[best_topic] > 0 else "other"

# Wrap tqdm around the data list
for convo in tqdm(data, desc="Classifying topics"):
    convo["topic"] = classify_topic(convo["first_tweet"])

with open("conversation_data_with_topics.json", "w") as f:
    json.dump(data, f, indent=2)
