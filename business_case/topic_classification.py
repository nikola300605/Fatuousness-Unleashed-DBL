import json
from tqdm import tqdm
from transformers import pipeline

classifier = pipeline("zero-shot-classification", 
                     model="facebook/bart-large-mnli",) 


def classify_topics(data, candidate_labels = [
        "flight delay or cancellation",
        "luggage and baggage issues", 
        "booking and reservation problems",
        "customer service complaints"],threshold=0.5):
    """
    Perform the actual classification of topics in the data.
    """

    print("Starting zero-shot classification...")
    confidence_scores = []
 
    all_texts = []
    tweet_refs = []
    for convo in tqdm(data, desc="Gathering data"):
        text= convo["first_tweet"]

        # Truncate very long texts (BART has token limits)
        if len(text) > 500:
            text = text[:500]

        all_texts.append(text)
        tweet_refs.append(convo)

    results = classifier(all_texts, candidate_labels, batch_size=32)

    for convo, result in tqdm(zip(tweet_refs, results), desc="Classifying with BART", total=len(tweet_refs)):
        text = convo["first_tweet"]

        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        label = ""

        if top_score < threshold:
            label = "other"
        else:
            label = top_label

        
        # Map descriptive labels back to simple categories if needed
        label_mapping = {
            "flight delay or cancellation": "delay",
            "luggage and baggage issues": "luggage", 
            "booking and reservation problems": "booking",
            "customer service complaints": "customer_service",
            "other": "other"
        }
        
        convo["topic"] = label_mapping.get(label, label)
        convo["topic_confidence"] = top_score
        confidence_scores.append(top_score)

    # Print statistics
    print(f"\nClassification complete!")
    print(f"Average confidence: {sum(confidence_scores)/len(confidence_scores):.3f}")
    print(f"Low confidence predictions (<0.5): {sum(1 for s in confidence_scores if s < 0.5)}")

    # Topic distribution
    topic_counts = {}
    for convo in data:
        topic = convo["topic"]
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    print("\nTopic Distribution:")
    for topic, count in sorted(topic_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / len(data)) * 100
        print(f"{topic}: {count} ({percentage:.1f}%)")

    # Save results
    with open("conversation_data_with_topics_zeroshot.json", "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to conversation_data_with_topics_zeroshot.json")

if __name__ == "__main__":
    with open("conversation_data_cleaned.json") as f:
        data = json.load(f)

    classify_topics(data)