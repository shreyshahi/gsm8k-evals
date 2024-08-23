import random
from datasets import load_dataset
from techniques.base import BaseTechnique, BaseWithBreaksTechnique, StraightToAnswer
from openai import OpenAI
import os
import json
from tqdm import tqdm

# Load the GSM8K dataset
dataset = load_dataset("openai/gsm8k", "main")

# Select 10 random problems from the training set
train_data = dataset["train"]
random_indices = random.sample(range(len(train_data)), 10)
selected_problems = [train_data[i] for i in random_indices]

# Initialize OpenAI client (make sure to set your API key in the environment)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def get_response(prompt, model):
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Evaluation function
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
    # Evaluate each technique
    results = {}

    # Initialize the techniques
    techniques = [
        ("BaseTechnique", BaseTechnique()),
        ("BaseWithBreaksTechnique", BaseWithBreaksTechnique()),
        ("StraightToAnswer", StraightToAnswer())
    ]

    # Initialize the models
    models = ["gpt-4o", "gpt-4o-mini"]

    # Create traces folder if it doesn't exist
    os.makedirs("traces", exist_ok=True)

    for model in models:
        for technique_name, technique in techniques:
            print(f"\nEvaluating {technique_name} with model {model}...")
            correct_count = 0
            technique_traces = []
            
            for problem in tqdm(selected_problems):
                is_correct, traces = evaluate_technique(technique, model, problem)
                if is_correct:
                    correct_count += 1
                technique_traces.append({
                    "problem": problem,
                    "traces": traces,
                    "is_correct": is_correct
                })
            
            accuracy = correct_count / len(selected_problems)
            results[f"{technique_name}_{model}"] = {"accuracy": accuracy}
            
            print(f"{technique_name} with model {model} Accuracy: {accuracy:.2f}")

            # Save traces for this technique and model
            technique_filename = f"traces/{technique_name.replace(' ', '_').lower()}_{model.replace('-', '_')}_traces.json"
            with open(technique_filename, "w") as f:
                json.dump(technique_traces, f, indent=2)

    # Save overall results
    with open("traces/evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation complete. Results saved in 'traces/evaluation_results.json'")
    print("Individual technique traces saved in the 'traces' folder.")

if __name__ == "__main__":
    main()
