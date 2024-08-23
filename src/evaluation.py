import random
from datasets import load_dataset
from techniques.base import BaseTechnique, StraightToAnswer
import openai
import os
import json
from tqdm import tqdm
import time

client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_response(prompt, model):
    max_retries = 5
    base_delay = 1
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except openai.RateLimitError as e:
            if attempt == max_retries - 1:
                raise e
            delay = base_delay * (2 ** attempt)
            time.sleep(delay)
        except Exception as e:
            raise e

def evaluate_technique(technique, model, problem):
    question = problem["question"]
    correct_answer = problem["answer"].split("###")[-1]
    correct_answer = "".join(c for c in correct_answer.strip() if c.isdigit() or c == ".")
    
    initial_prompt = technique.get_initial_prompt(question)
    traces = []
    
    while not technique.is_complete(question, [trace["response"] for trace in traces]):
        if not traces:
            prompt = initial_prompt
        else:
            prompt = technique.get_follow_up_prompt(question, traces)
            if prompt is None:
                break
        
        response = get_response(prompt, model)
        traces.append({"prompt": prompt, "response": response})
    
    # Extract numerical answer using a smaller LLM
    final_answer = extract_numerical_answer(traces[-1]["response"])
    
    is_correct = compare_answers(final_answer, correct_answer)
    
    return is_correct, traces

def extract_numerical_answer(text_answer):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Extract the final numerical answer from the given text. Respond with only the final numerical answer."},
            {"role": "user", "content": text_answer}
        ]
    )
    return ''.join(c for c in response.choices[0].message.content.strip() if c.isdigit() or c == '.')

def compare_answers(extracted_answer, correct_answer):
    return str(extracted_answer).strip() == str(correct_answer).strip()

def main():
    # Load the GSM8K dataset
    dataset = load_dataset("openai/gsm8k", "main")

    # Select 500 random problems from the training set
    train_data = dataset["train"]

    random_indices_file = "selected_problems.json"
    if os.path.exists(random_indices_file):
        with open(random_indices_file, "r") as f:
            selected_indices = json.load(f)
    else:
        selected_indices = random.sample(range(len(train_data)), 500)
        with open(random_indices_file, "w") as f:
            json.dump(selected_indices, f)

    selected_problems = [train_data[i] for i in selected_indices]

    # Evaluate each technique
    results = {}

    # Initialize the techniques
    techniques = [
        ("BaseTechnique", BaseTechnique()),
        ("StraightToAnswer", StraightToAnswer())
    ]

    # Initialize the models
    models = ["gpt-4o", "gpt-4o-mini"]

    # Create traces folder if it doesn't exist
    os.makedirs("traces", exist_ok=True)

    for model in models:
        for technique_name, technique in techniques:
            print(f"\nEvaluating {technique_name} with model {model}...")
            technique_filename = f"traces/{technique_name.replace(' ', '_').lower()}_{model.replace('-', '_')}_traces.json"
            
            if os.path.exists(technique_filename):
                print(f"Traces file already exists for {technique_name} with model {model}. Skipping evaluation.")
                with open(technique_filename, "r") as f:
                    technique_traces = json.load(f)
                correct_count = sum(1 for trace in technique_traces if trace["is_correct"])
            else:
                correct_count = 0
                technique_traces = []
                
                for i, problem in enumerate(tqdm(selected_problems, desc=f"{technique_name} - {model}", leave=False)):
                    is_correct, traces = evaluate_technique(technique, model, problem)
                    if is_correct:
                        correct_count += 1
                    technique_traces.append({
                        "problem": problem,
                        "traces": traces,
                        "is_correct": is_correct
                    })
                    accuracy_so_far = correct_count / (i + 1)
                    tqdm.write(f"Progress: {i+1}/{len(selected_problems)} | Correct: {correct_count} | Accuracy: {accuracy_so_far:.2%}")
                    time.sleep(1)  # Add a small delay to avoid rate limiting
                
                # Save traces for this technique and model
                with open(technique_filename, "w") as f:
                    json.dump(technique_traces, f, indent=2)
            
            accuracy = correct_count / len(selected_problems)
            results[f"{technique_name}_{model}"] = {"accuracy": accuracy}
            
            print(f"{technique_name} with model {model} Accuracy: {accuracy:.2f}")
            time.sleep(5)  # Add a longer delay between technique/model combinations

    # Save overall results
    with open("traces/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete. Results saved in 'traces/evaluation_results.json'")
    print("Individual technique traces saved in the 'traces' folder.")

if __name__ == "__main__":
    main()
