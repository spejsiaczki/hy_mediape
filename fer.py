import cv2
from fer import FER

# input_img = cv2.imread("test.png")
input_video = cv2.VideoCapture("videos/HY_2024_film_01.mp4")
frame_rate = input_video.get(cv2.CAP_PROP_FPS)
emotion_detector = FER()
emotions_dict = {'angry':[], 'disgust':[], 'fear':[], 'happy':[], 'sad':[], 'surprise':[], 'neutral':[]}

while input_video.isOpened():
    success, frame = input_video.read()
    if not success:
        print('dupa')
        break


    result = emotion_detector.detect_emotions(frame)
    if not result:
        # print('dupa')
        continue
    print(result)
    # bounding_box = result[0]['box']
    emotions = result[0]['emotions']
    # cv2.rectangle(
    #     input_img, 
    #     (bounding_box[0], bounding_box[1]), 
    #     (bounding_box[0]+bounding_box[2], bounding_box[1]+bounding_box[3]), 
    #     (0,0,255), 
    #     2)
    
    for emotion, percent in emotions.items():
        emotions_dict[str(emotion)].append(percent)
    # if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
    #     break
input_video.release()

cv2.destroyAllWindows()

with open('data.txt', 'w') as file:
    file.write(str(emotions_dict))