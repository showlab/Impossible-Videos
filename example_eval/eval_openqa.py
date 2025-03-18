import os
import json
import torch
from tqdm import tqdm
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from glob import glob
from gpt4o_evalutor import generate_score


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
        return "The video is plausible without any impossible events."
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
    for video_name, question in tqdm(input_data.items()):
        video_file = get_video_path(video_name, data_path)
        result = inference_one(video_file, question)
        pred_dict[video_name] = {'video_name': video_name, 'pred': result}

    with open(pred_file, 'w') as f:
        json.dump(pred_dict, f)


def compute_overall_score(output_dir, gt_file, data_path):
    with open(f"{data_path}/video2taxonomy_label.json", 'r') as f:
        vid_to_tax = json.load(f)

    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    data_input = {}
    list_files = glob(os.path.join(output_dir, "*.txt"))
    for file in list_files:
        vid_name = os.path.basename(file).replace('.txt', '')
        data = open(file, 'r').read()
        data_input[vid_name] = data

    assert len(data_input) == len(gt_data)

    accumu_score = 0.0
    accumu_phy, accumu_bio, accumu_social, accumu_geo = 0.0, 0.0, 0.0, 0.0
    accumu_spa, accumu_tmp = 0.0, 0.0
    cnt_phy, cnt_bio, cnt_social, cnt_geo = 0, 0, 0, 0
    cnt_spa, cnt_tmp = 0, 0
    for k, v in data_input.items():

        if 'physical laws' in vid_to_tax[k]['taxonomy_label_list']:
            cnt_phy += 1
        if 'biological laws' in vid_to_tax[k]['taxonomy_label_list']:
            cnt_bio += 1
        if 'social laws' in vid_to_tax[k]['taxonomy_label_list']:
            cnt_social += 1
        if 'geographical laws' in vid_to_tax[k]['taxonomy_label_list']:
            cnt_geo += 1

        if vid_to_tax[k]['spatial_temporal_label'] == "spatial":
            cnt_spa += 1
        elif vid_to_tax[k]['spatial_temporal_label'] == "temporal":
            cnt_tmp += 1
        else:
            raise ValueError

        json_str = str(v).replace("json", "").replace("```", '')
        try:
            data = json.loads(json_str)
        except Exception as e:
            print(k)
            print(json_str)
            continue
        data['semantic_alignment_score'] = str(data['semantic_alignment_score'])
        if len(data['semantic_alignment_score']) > 4:
            assert '-' in data['semantic_alignment_score'], data['semantic_alignment_score']
            lower = float(data['semantic_alignment_score'].split('-')[0])
            upper = float(data['semantic_alignment_score'].split('-')[1])
            cur_score = (lower + upper) / 2.0
            print("Averaging {} and {} into {}".format(lower, upper, (lower + upper) / 2.0))
        else:
            cur_score = float(data['semantic_alignment_score'])

        accumu_score += cur_score
        if 'physical laws' in vid_to_tax[k]['taxonomy_label_list']:
            accumu_phy += cur_score
        if 'biological laws' in vid_to_tax[k]['taxonomy_label_list']:
            accumu_bio += cur_score
        if 'social laws' in vid_to_tax[k]['taxonomy_label_list']:
            accumu_social += cur_score
        if 'geographical laws' in vid_to_tax[k]['taxonomy_label_list']:
            accumu_geo += cur_score

        if vid_to_tax[k]['spatial_temporal_label'] == "spatial":
            accumu_spa += cur_score
        elif vid_to_tax[k]['spatial_temporal_label'] == "temporal":
            accumu_tmp += cur_score
        else:
            raise ValueError

    # assert cnt_spa + cnt_tmp == len(data_input)
    print("Overall score: {:.1f}".format(accumu_score / len(data_input)*100))
    print("Physical score: {:.1f}".format(accumu_phy / cnt_phy*100))
    print("Biological score: {:.1f}".format(accumu_bio / cnt_bio*100))
    print("Social score: {:.1f}".format(accumu_social / cnt_social*100))
    print("Geographical score: {:.1f}".format(accumu_geo / cnt_geo*100))
    print("Spatial score: {:.1f}".format(accumu_spa / cnt_spa*100))
    print("Temporal score: {:.1f}".format(accumu_tmp / cnt_tmp*100))
    print("=" * 50)


if __name__ == '__main__':
    # Step 0: config the path
    data_path = "/users/zechen/ImpV/arxiv/release"
    question_file = f"{data_path}/openqa_question.json"
    answer_file = f"{data_path}/openqa_answer.json"

    # Step 1: config the model name
    model_name = "qwen2_vl"
    pred_file = f"{model_name}_pred_ipv_openqa.json"
    output_score_folder = f"{model_name}_openqa_score"

    # Step 2: run inference
    main_proc(question_file, pred_file, data_path)

    # Step 3: run GPT-4o score evaluation
    generate_score(answer_file, pred_file, output_score_folder)

    # Step 4: compute final score
    compute_overall_score(output_score_folder, answer_file, data_path)

