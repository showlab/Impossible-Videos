import os
import json
import torch
import random
from tqdm import tqdm


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
    return os.path.join(data_path, "impossible_videos", video_name)


def inference_one(video_file, question):
    if RANDOM_TEST:
        return random.choice(['A', 'B', 'C', 'D', 'E'])
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


def compute_accuracy_mcqa(pred_file, gt_file, data_path):
    with open(f"{data_path}/video2taxonomy_label.json", 'r') as f:
        vid_to_tax = json.load(f)
    with open(pred_file, 'r') as f:
        pred_data = json.load(f)
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)
    assert len(pred_data) == len(gt_data)

    num_total = len(pred_data)
    num_correct = 0

    cnt_phy, cnt_bio, cnt_social, cnt_geo = 0, 0, 0, 0
    cnt_spa, cnt_tmp = 0, 0

    correct_phy, correct_bio, correct_social, correct_geo = 0, 0, 0, 0
    correct_spa, correct_tmp = 0, 0

    for question_id in pred_data.keys():
        pred = str(pred_data[question_id]['pred']).lower().replace('.', '').replace('(', '').replace(')', '').strip()[:1]
        gt_ans = str(gt_data[question_id]['answer']).lower().strip()

        if 'physical laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
            cnt_phy += 1
        if 'biological laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
            cnt_bio += 1
        if 'social laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
            cnt_social += 1
        if 'geographical laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
            cnt_geo += 1

        if vid_to_tax[gt_data[question_id]['video_name']]['spatial_temporal_label'] == "spatial":
            cnt_spa += 1
        elif vid_to_tax[gt_data[question_id]['video_name']]['spatial_temporal_label'] == "temporal":
            cnt_tmp += 1
        else:
            raise ValueError

        if pred == gt_ans:
            num_correct += 1
            if 'physical laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
                correct_phy += 1
            if 'biological laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
                correct_bio += 1
            if 'social laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
                correct_social += 1
            if 'geographical laws' in vid_to_tax[gt_data[question_id]['video_name']]['taxonomy_label_list']:
                correct_geo += 1

            if vid_to_tax[gt_data[question_id]['video_name']]['spatial_temporal_label'] == "spatial":
                correct_spa += 1
            if vid_to_tax[gt_data[question_id]['video_name']]['spatial_temporal_label'] == "temporal":
                correct_tmp += 1

    assert cnt_phy + cnt_bio + cnt_social + cnt_geo > num_total

    print("Num total: {}".format(num_total))
    print("Num correct: {}".format(num_correct))
    print("Num total: {}".format(num_total))
    print("Accuracy is {:.1f}".format(num_correct/num_total*100))
    print("Physical accuracy: {:.1f}".format(correct_phy / cnt_phy*100))
    print("Biological accuracy: {:.1f}".format(correct_bio / cnt_bio*100))
    print("Social accuracy: {:.1f}".format(correct_social / cnt_social*100))
    print("Geographical accuracy: {:.1f}".format(correct_geo / cnt_geo*100))
    print("Spatial accuracy: {:.1f}".format(correct_spa / cnt_spa*100))
    print("Temporal accuracy: {:.1f}".format(correct_tmp / cnt_tmp*100))
    print("=" * 50)


if __name__ == '__main__':
    # Step 0: config the path
    data_path = "/users/zechen/ImpV/arxiv/release"
    question_file = f"{data_path}/mcqa_question.json"
    answer_file = f"{data_path}/mcqa_answer.json"

    # Step 1: config the model name
    model_name = "qwen2_vl"
    pred_file = f"{model_name}_pred_ipv_mcqa.json"

    # Step 2: run inference
    print("Evaluating model {} on the MCQA dataset...".format(model_name))
    main_proc(question_file, pred_file, data_path)

    # Step 3: compute accuracy
    compute_accuracy_mcqa(pred_file, answer_file, data_path)

