prompt_template = '''You are tasked with evaluating the semantic similarity between a model-generated answer and the ground-truth answers. Your goal is to determine how well the prediction aligns with the intended meaning of the ground-truth answers based on high-level semantic understanding.

### **Input Provided**:

1. **Ground-Truth Answers**:

   - Primary Answer: [Primary answer]
   - Alternative Answers:
     - [Alternative answer 1]
     - [Alternative answer 2]
     - [Alternative answer 3]

2. **Model's Prediction**:

   - [Model's prediction]

### **Task**:

Evaluate whether the model's prediction aligns with the meaning of the ground-truth answers. Specifically:

1. Does the prediction capture the core idea of the counterintuitive or impossible phenomena?
2. Is the prediction consistent with the ground-truth answers in meaning?
3. Is the prediction accurate, relevant, and natural?
4. Normalize for length differences. If the prediction is longer or shorter than the ground-truth, focus on whether it captures the core semantic meaning of the counterintuitive phenomenon without being penalized for verbosity or brevity.

### **Evaluation Criteria**:

- Justify the score:

  - Highlight key matches or mismatches between the prediction and ground-truth.
  - Mention whether the prediction introduced irrelevant or incorrect information.

- Assign a semantic alignment score between 0 and 1:

  - **1.0**: Perfect alignment (prediction fully matches the meaning of the ground-truth answers).
  - **0.8-0.9**: Good alignment (prediction captures the main idea but may slightly vary in expression or include minor irrelevant details).
  - **0.5-0.7**: Partial alignment (prediction captures some aspects but misses important details or adds unrelated information).
  - **0.1-0.4**: Weak alignment (prediction is somewhat relevant but largely incorrect, incomplete, or includes significant unrelated content).
  - **0.0**: No alignment (prediction is irrelevant, incorrect, or completely off-topic).

### **Output Format**:

First, write the justification explaining the alignment between the prediction and the ground-truth. Then, based on the justification, assign a semantic alignment score. Provide your response in the following JSON format:

```
{
  "justification": "Brief explanation of why you assigned this score, mentioning any key matches or mismatches.",
  "semantic_alignment_score": "Score between 0 and 1"
}
```

### **Example Input and Output**

**Example 1**:

**Input**:

- **Ground-Truth Answers**:
  ```
  {
    "primary_answer": "The car floats upward instead of falling, defying gravity.",
    "alternative_answers": [
      "Instead of falling, the car floats upward, which violates gravity.",
      "The car defies the law of gravity by floating upward after driving off the cliff.",
      "The car floats upward rather than falling as expected, breaking the law of gravity."
    ]
  }
  ```
- **Model's Prediction**:
  "The car rises into the air instead of falling, which defies gravity."

**Output**:

```
{
  "justification": "The prediction captures the core phenomenon (the car rising instead of falling) and aligns well with the meaning of the ground-truth answers. It is accurate, relevant, and natural.",
  "semantic_alignment_score": 1.0
}
```

**Example 2**:

**Input**:

- **Ground-Truth Answers**:
  ```
  {
    "primary_answer": "The rock rolls uphill, defying gravity.",
    "alternative_answers": [
      "Instead of rolling downhill, the rock moves uphill, which violates gravity.",
      "The rock moves upward on the slope rather than downward, breaking the law of gravity."
    ]
  }
  ```
- **Model's Prediction**:
  "The rock moves upward on the slope, breaking gravity."

**Output**:

```
{
  "justification": "The prediction captures the main idea but slightly simplifies the explanation, missing the explicit comparison to rolling downhill.",
  "semantic_alignment_score": 0.9
}
```

**Example 3**:

**Input**:

- **Ground-Truth Answers**:
  ```
  {
    "primary_answer": "The ball bounces higher after each bounce, defying the laws of physics.",
    "alternative_answers": [
      "Instead of losing energy, the ball gains height with every bounce, breaking the laws of physics.",
      "The ball violates the laws of physics by bouncing higher after each impact."
    ]
  }
  ```
- **Model's Prediction**:
  "The ball keeps bouncing higher, which is unusual."

**Output**:

```
{
  "justification": "The prediction captures part of the phenomenon (bouncing higher) but lacks detail about defying the laws of physics and does not explicitly mention the gain in height after each bounce.",
  "semantic_alignment_score": 0.7
}
```

**Example 4**:

**Input**:

- **Ground-Truth Answers**:
  ```
  {
    "primary_answer": "The sun sets in the east and rises in the west, reversing the natural order.",
    "alternative_answers": [
      "Instead of setting in the west, the sun sets in the east and rises in the west.",
      "The sun's behavior is reversed, rising in the west and setting in the east."
    ]
  }
  ```
- **Model's Prediction**:
  "The sun rises in the west."

**Output**:

```
{
  "justification": "The prediction captures part of the phenomenon (sun rising in the west) but omits the reversal of the setting direction, making it incomplete.",
  "semantic_alignment_score": 0.5
}
```

NOTE: You directly output the result without explanation or other words.

Input:
'''

import time
import json
import os
from tqdm import tqdm
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

RANDOM_TEST = True
if RANDOM_TEST:
    print("Running in random test mode...")

client = OpenAI(
    api_key="YOUR_API_KEY",
)


def generate_message(text_prompt):
    if RANDOM_TEST:
        return '''
        {"semantic_alignment_score": 0.5,
         "justification": "Random test, no ground truth answers provided."}
        '''
    time.sleep(1)
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": text_prompt,
                    },
                ],
            }
        ],
    )

    output = str(response.choices[0].message.content).strip()

    return output


def generate_score(gt_file, pred_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    assert len(gt_data) == len(pred_data)

    output_dict = {}
    for vid_name in tqdm(gt_data.keys()):
        save_file = os.path.join(output_dir, vid_name + '.txt')
        if os.path.exists(save_file):
            print("Found {} already exists, skip and continue.".format(save_file))
            continue

        gt_answer = gt_data[vid_name]
        pred_answer = pred_data[vid_name]['pred']
        input_seq = '''Primary Answer: {primary_answer}
- Alternative Answers:
     - [alternative_answer_1]
     - [alternative_answer_2]
     - [alternative_answer_3]

Model's Prediction: {model_pred}
'''
        input_seq = input_seq.format(
            primary_answer=gt_answer['primary_answer'],
            alternative_answer_1=gt_answer['alternative_answers'][0],
            alternative_answer_2=gt_answer['alternative_answers'][1],
            alternative_answer_3=gt_answer['alternative_answers'][2],
            model_pred=pred_answer,
        )
        full_prompt = prompt_template + input_seq
        response = generate_message(full_prompt)
        output = str(response).strip()
        # print(output)

        if output is not None:
            output_dict[vid_name] = output
            with open(save_file, 'w') as f:
                f.write(output)
        else:
            print("Error reading {}, skip it now.".format(vid_name))
    with open(os.path.join(output_dir, "gpt_eval_global.json"), 'w') as f:
        json.dump(output_dict, f)

