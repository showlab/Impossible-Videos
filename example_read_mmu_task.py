import os
import json


def sanity_check():
    # ========== Judgement Task ==========
    with open("judgement_question.json", "r") as f:
        judgement_question_json = json.load(f)
    with open("judgement_answer.json", "r") as f:
        judgement_answer_json = json.load(f)
    assert len(judgement_question_json) == len(judgement_answer_json)
    print(f"Judgement Task: {len(judgement_question_json)} questions")
    for question_id in judgement_question_json:
        assert question_id in judgement_answer_json
        video_name = judgement_question_json[question_id]["video_name"]
        assert os.path.exists(f"impossible_videos/{video_name}") or os.path.exists(f"real_world_videos/{video_name}")

    # ========== Multi-choice QA Task ==========
    with open("mcqa_question.json", "r") as f:
        mcqa_question_json = json.load(f)
    with open("mcqa_answer.json", "r") as f:
        mcqa_answer_json = json.load(f)
    assert len(mcqa_question_json) == len(mcqa_answer_json)
    print(f"Multi-choice Task: {len(mcqa_question_json)} questions")
    for question_id in mcqa_question_json:
        assert question_id in mcqa_answer_json
        video_name = mcqa_question_json[question_id]["video_name"]
        assert os.path.exists(f"impossible_videos/{video_name}")

    # ========== Open-ended QA Task ==========
    with open("openqa_question.json", "r") as f:
        openqa_question_json = json.load(f)
    with open("openqa_answer.json", "r") as f:
        openqa_answer_json = json.load(f)
    assert len(openqa_question_json) == len(openqa_answer_json)
    print(f"Open-ended Task: {len(openqa_question_json)} questions")
    for question_id in openqa_question_json:
        assert question_id in openqa_answer_json
        video_name = question_id
        assert os.path.exists(f"impossible_videos/{video_name}")


if __name__ == '__main__':
    sanity_check()

