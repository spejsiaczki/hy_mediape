import cv2
from fer import FER
import sys


def save_emotions_data(filename:str, data:dict):
    with open(f'emotions_detection/{filename}.txt', 'w') as file:
        file.write(str(data))

def get_filename(input_file:str)->str:
    return input_file.replace(".mp4","_data")

def main(videos_path:str, file_path:str):
    input_video = cv2.VideoCapture(videos_path+file_path)
    frame_rate = input_video.get(cv2.CAP_PROP_FPS)
    emotion_detector = FER()
    emotions_dict = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}

    while input_video.isOpened():
        success, single_frame = input_video.read()
        if not success:
            break
        result = emotion_detector.detect_emotions(single_frame)
        if not result:
            continue
        emotions = result[0]['emotions']
        for emotion, percent in emotions.items():
            emotions_dict[str(emotion)].append(percent)

    input_video.release()
    cv2.destroyAllWindows()

    filename = get_filename(file_path)
    save_emotions_data(filename, emotions_dict)

if __name__=="__main__":
    videos_path = "/home/probook/Documents/repos/hy_mediape/videos/"
    # main(f"{videos_path}/HY_2024_film_08.mp4")
    main(videos_path, "HY_2024_film_08.mp4")
    # print(get_filename("videos/HY_2024_film_01.mp4"))