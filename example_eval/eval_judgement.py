import os
import json
import torch
import random
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score


RANDOM_TEST = True

if not RANDOM_TEST:
    from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
    from qwen_vl_utils import process_vision_info
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
else:
    print("Running in random test mode...")


def get_video_path(video_name, data_path):
    if os.path.exists(os.path.join(data_path, "impossible_videos", video_name)):
        return os.path.join(data_path, "impossible_videos", video_name)
    return os.path.join(data_path, "real_world_videos", video_name)


def is_real_video(video_name, data_path):
    if os.path.exists(os.path.join(data_path, "impossible_videos", video_name)):
        return False
    return True


def inference_one(video_file, question):
    if RANDOM_TEST:
        return random.choice(['yes', 'no'])

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "file://{}".format(video_file),
                },
                {"type": "text",
                 "text": question},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    output_text = output_text[0].strip()
    print(output_text)
    return output_text


def main_proc(question_file, pred_file, data_path):
    with open(question_file, 'r') as f:
        input_data = json.load(f)

    pred_dict = {}
    for question_id, question_dict in tqdm(input_data.items()):
        video_name = question_dict['video_name']
        video_file = get_video_path(video_name, data_path)
        question = question_dict['question']
        result = inference_one(video_file, question)
        pred_dict[question_id] = {'video_name': video_name, 'pred': result}

    with open(pred_file, 'w') as f:
        json.dump(pred_dict, f)


def compute_accuracy(pred_file, answer_file, data_path):
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with open(answer_file, 'r') as f:
        gt_data = json.load(f)
    assert len(pred_data) == len(gt_data)

    num_total = len(pred_data)
    num_correct = 0
    cnt_fake, cnt_real = 0, 0
    correct_fake, correct_real = 0, 0
    for question_id in pred_data.keys():
        if question_id not in gt_data.keys():
            continue
        if is_real_video(pred_data[question_id]['video_name'], data_path):
            cnt_real += 1
        else:
            cnt_fake += 1
        pred = str(pred_data[question_id]['pred']).lower().replace('.', '').replace('(', '').replace(')', '').strip()
        pred = pred[:3].replace(',', '').strip()
        gt_ans = str(gt_data[question_id]['answer']).lower().strip()
        if pred == gt_ans:
            num_correct += 1
            if is_real_video(pred_data[question_id]['video_name'], data_path):
                correct_real += 1
            else:
                correct_fake += 1
    assert num_total == len(gt_data)
    assert cnt_real + cnt_fake == num_total
    assert correct_real + correct_fake == num_correct
    print("Total number of questions: ", num_total)
    print("Accuracy is {:.1f}".format(num_correct/num_total*100))
    print("Accuracy on fake videos: {:.1f}".format(correct_fake / cnt_fake * 100))
    print("Accuracy on real videos: {:.1f}".format(correct_real / cnt_real * 100))
    print("--"*50)


def compute_accuracy_Fscore(pred_file, answer_file):
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with open(answer_file, 'r') as f:
        gt_data = json.load(f)
    assert len(pred_data) == len(gt_data)

    y_true = []
    y_pred = []

    for question_id in pred_data.keys():
        if question_id not in gt_data.keys():
            continue
        pred = str(pred_data[question_id]['pred']).lower().replace('.', '').replace('(', '').replace(')', '').strip()
        pred = pred[:3].replace(',', '').strip()
        gt_ans = str(gt_data[question_id]['answer']).lower().strip()

        if pred == "yes":
            y_pred.append(1)
        elif pred == "no":
            y_pred.append(0)
        else:
            print(pred)
            continue

        if gt_ans == "yes":
            y_true.append(1)
        elif gt_ans == "no":
            y_true.append(0)
        else:
            raise NotImplementedError

    # 计算准确率
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.1f}')

    # 计算F1分数
    f_score = f1_score(y_true, y_pred)
    print(f'F1 Score: {f_score*100:.1f}')

    yes_rate = sum(y_pred) / len(y_pred)
    print(f'Yes rate: {yes_rate*100:.1f}')


if __name__ == '__main__':
    # Step 0: config the path
    data_path = "/users/zechen/ImpV/arxiv/release"
    question_file = f"{data_path}/judgement_question.json"
    answer_file = f"{data_path}/judgement_answer.json"

    # Step 1: config the model name
    model_name = "qwen2_vl"
    pred_file = f"{model_name}_pred_ipv_judgement.json"

    # Step 2: run inference
    print("Evaluating model {} on the judgement dataset...".format(model_name))
    main_proc(question_file, pred_file, data_path)

    # Step 3: compute accuracy and F-score
    compute_accuracy(pred_file, answer_file, data_path)
    compute_accuracy_Fscore(pred_file, answer_file)

