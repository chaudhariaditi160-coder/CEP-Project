from transformers import pipeline

qa_pipeline = pipeline("question-answering")

def get_answer(question, context):
    result = qa_pipeline(question=question, context=context)

    # Confidence check
    if result['score'] < 0.3:
        return "Answer not found in the document."

    return result['answer']
