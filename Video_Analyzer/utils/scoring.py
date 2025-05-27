from sentence_transformers import util

from models.qa_model import load_qa_model
model = load_qa_model()


def evaluate_technical_answers_with_explanation(transcript,qa_set):
    sentences = [s.strip() for s in transcript.lower().split('.') if s.strip()]
    resuls = []
    total_score = 0
    
    for question, expected_answer in qa_set.items():
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        question_embedding = model.encode(question,convert_to_tensor=True)
        
        similarities = util.cos_sim(question_embedding, sentence_embeddings)[0]
        best_idx = similarities.argmax().item()
        best_sentence = sentences[best_idx]
        
        best_sentence_embedding = model.encode(best_sentence, convert_to_tensor=True)
        expected_embedding = model.encode(expected_answer, convert_to_tensor=True)
        answer_similarity = util.cos_sim(best_sentence_embedding, expected_embedding).item()
        
        score = round(answer_similarity*10,2)
        total_score+=score
        
        resuls.append({
            "Question": question,
            "Expected Answer": expected_answer,
            "Best Match from Response": best_sentence,
            "Score": score
        })
        
    average_score = round(total_score/ len(qa_set),2) if qa_set else 0.0
    return resuls, average_score
