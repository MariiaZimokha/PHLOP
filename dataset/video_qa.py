import cv2
import json
import numpy as np
import random


def add_qa_to_video(questions_answers, object_json_file, output_path, num_objects):
    with open(object_json_file, 'r') as f:
        sim_data = json.load(f)

    video_path = f"{output_path}output_video.mp4"
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("FPS is 0, using fallback FPS of 25.")
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_with_qa_path = f'{output_path}output_with_qa.mp4'
    out = cv2.VideoWriter(output_with_qa_path, fourcc, fps, (width, height))

    last_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        last_frame = frame.copy()
        out.write(frame)
    cap.release()

    sample_qa = (
        questions_answers if len(questions_answers) < 5
        else random.sample(questions_answers, 5)
    )

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    font_color = (255, 255, 255)
    thickness = 2
    shadow_color = (0, 0, 0)
    shadow_offset = (2, 2)
    background_color = (0, 0, 0)
    background_alpha = 0.6
    line_type = cv2.LINE_AA
    y0 = 50
    dy = 40
    max_width = width - 100

    def wrap_text(text, max_width, font_face, font_scale, thickness):
        words = text.split(' ')
        lines = []
        current_line = ''
        for word in words:
            test_line = current_line + word + ' '
            text_size = cv2.getTextSize(test_line, font_face, font_scale, thickness)[0]
            if text_size[0] <= max_width:
                current_line = test_line
            else:
                lines.append(current_line.strip())
                current_line = word + ' '
        lines.append(current_line.strip())
        return lines

    cumulative_text = []
    qa_frames = []

    for qa in sample_qa:
        question = qa["question"]
        answer = qa["answer"]

        cumulative_text.append("Q: " + question)
        text_block = "\n\n".join(cumulative_text)
        frame_text = last_frame.copy()

        num_lines = len(text_block.split('\n'))
        text_height = num_lines * dy + (num_lines - 1) * 10
        background_rect = np.zeros((text_height, width, 3), dtype=np.uint8)
        background_rect[:] = background_color
        background_rect = cv2.addWeighted(background_rect, background_alpha,
                                          np.zeros_like(background_rect), 1 - background_alpha, 0)

        y0_start = y0 - 10
        frame_text[y0_start:y0_start + text_height, 0:width] = cv2.addWeighted(
            frame_text[y0_start:y0_start + text_height, 0:width], 1 - background_alpha,
            background_rect, background_alpha, 0)

        y = y0
        for paragraph in text_block.split('\n\n'):
            wrapped_lines = wrap_text(paragraph, max_width, font, font_scale, thickness)
            for line in wrapped_lines:
                cv2.putText(frame_text, line,
                            (50 + shadow_offset[0], y + shadow_offset[1]),
                            font, font_scale, shadow_color, thickness, line_type)
                cv2.putText(frame_text, line,
                            (50, y), font, font_scale, font_color, thickness, line_type)
                y += dy
            y += dy

        qa_frames.append(frame_text.copy())

        cumulative_text[-1] = "Q: " + question + "\nA: " + answer
        text_block = "\n\n".join(cumulative_text)
        frame_text = last_frame.copy()

        num_lines = len(text_block.split('\n'))
        text_height = num_lines * dy + (num_lines - 1) * 10
        background_rect = np.zeros((text_height, width, 3), dtype=np.uint8)
        background_rect[:] = background_color
        background_rect = cv2.addWeighted(background_rect, background_alpha,
                                          np.zeros_like(background_rect), 1 - background_alpha, 0)

        y0_start = y0 - 10
        frame_text[y0_start:y0_start + text_height, 0:width] = cv2.addWeighted(
            frame_text[y0_start:y0_start + text_height, 0:width], 1 - background_alpha,
            background_rect, background_alpha, 0)

        y = y0
        for paragraph in text_block.split('\n\n'):
            wrapped_lines = wrap_text(paragraph, max_width, font, font_scale, thickness)
            for line in wrapped_lines:
                cv2.putText(frame_text, line,
                            (50 + shadow_offset[0], y + shadow_offset[1]),
                            font, font_scale, shadow_color, thickness, line_type)
                cv2.putText(frame_text, line,
                            (50, y), font, font_scale, font_color, thickness, line_type)
                y += dy
            y += dy

        qa_frames.append(frame_text.copy())

    frames_per_display = int(fps * 1)
    for qa_frame in qa_frames:
        for _ in range(frames_per_display):
            out.write(qa_frame)

    out.release()
    # print(f"Output video with QAs saved at: {output_with_qa_path}")
