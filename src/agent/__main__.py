from .intent_extractor import extract_intent_with_llm

if __name__ == "__main__":
    query = "Show me the temperature trend in the Indian Ocean for the last year"
    intent = extract_intent_with_llm(query)
    print("\n🔍 Extracted Intent:")
    print(intent.model_dump_json(indent=2))
