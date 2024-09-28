import cv2
from fer import FER
import numpy as np


def save_emotions_data(filename:str, data:dict):
    with open(f'emotions_detection/{filename}_test.txt', 'w') as file:
        file.write(str(data))

def get_filename(input_file:str)->str:
    return input_file.replace(".mp4","_data")

def main(videos_path:str, file_path:str):
    input_video = cv2.VideoCapture(videos_path+file_path)
    frame_rate = input_video.get(cv2.CAP_PROP_FPS)
    frame_idx = 0

    emotion_detector = FER()
    emotions_dict = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}
    emotions_labels = list(emotions_dict.keys())
    # emotion_history = []
    # threshold = 0.1
    previous_value = 0
    found_zboczes = 0
    max_emotions_dict = {}

    while input_video.isOpened():
        emotions_in_segment = []
        emotion_history = []
        success, single_frame = input_video.read()
        if not success:
            break
        result = emotion_detector.detect_emotions(single_frame)
        if not result:
            continue
        emotions = result[0]['emotions']
        for emotion, percent in emotions.items():
            if frame_idx>0:
                previous_value = emotions_dict[str(emotion)][frame_idx-1]
            diff = percent - previous_value
            # print(emotion, previous_value, percent)
            if diff>0.4 and emotion!='neutral':
                # print(f"Frame {frame_idx} DUPA IN {emotion}, {previous_value}, {percent}")
                found_zboczes+=1
            emotions_dict[str(emotion)].append(percent)
            emotions_in_segment = emotions_dict
            # previous_value = percent
            
        
        if frame_idx%frame_rate==0:
            for key, value in emotions_in_segment.items():
                mean_emotion_values = np.mean(value)
                emotion_history.append(mean_emotion_values)
            # print(frame_idx)
            # print(emotion_history)
            max_emotion = emotions_labels[np.argmax(emotion_history)]
            # print(f'For second {frame_idx/frame_rate}, max emotion:{max_emotion}')
            max_emotions_dict[frame_idx/frame_rate] = max_emotion

        frame_idx+=1

    input_video.release()
    cv2.destroyAllWindows()

    filename = get_filename(file_path)
    save_emotions_data(filename, emotions_dict)
    return found_zboczes, max_emotions_dict

if __name__=="__main__":
    videos_path = "/home/probook/Documents/repos/hy_mediape/videos/"
    # main(f"{videos_path}/HY_2024_film_08.mp4")
    found_zboczes, max_emotions_dict = main(videos_path, "HY_2024_film_07.mp4")
    print(found_zboczes)
    if found_zboczes>3:
        print("Znaleziono błąd: mimika")
    print("Emocje:")
    for keys, values in max_emotions_dict.items():
        if int(keys)==0:
            print(f"Sekunda {keys}, emocja {values}")
            prev_emotion = values
        else:
            if values!=prev_emotion:
                print(f"Sekunda {keys}, emocja {values}")
                prev_emotion = values
    # unique_emotions = set(max_emotions_dict.values())
    # print(unique_emotions)

    # print(get_filename("videos/HY_2024_film_01.mp4"))