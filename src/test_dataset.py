# # import os

# # path = "data/Radvess"

# # print("Exists:", os.path.exists(path))

# # print("\nFirst folders:")
# # print(os.listdir(path)[:5])

# # import os

# # count = 0

# # for root, dirs, files in os.walk("data/Radvess"):
# #     for file in files:
# #         print(file)

# #         count += 1

# #         if count == 10:
# #             break

# #     if count == 10:
# #         break
    
    
# filename = "03-01-05-01-01-01-12.wav"

# emotion_code = filename.split("-")[2]

# print(emotion_code)

# emotion_map = {
#     "01": "neutral",
#     "02": "calm",
#     "03": "happy",
#     "04": "sad",
#     "05": "angry",
#     "06": "fear",
#     "07": "disgust",
#     "08": "surprise"
# }

# filename = "03-01-05-01-01-01-12.wav"

# emotion = emotion_map[filename.split("-")[2]]

# print(emotion)

# import os
# from collections import Counter

# emotion_map = {
#     "01": "neutral",
#     "02": "calm",
#     "03": "happy",
#     "04": "sad",
#     "05": "angry",
#     "06": "fear",
#     "07": "disgust",
#     "08": "surprise"
# }

# emotion_counter = Counter()

# for root, dirs, files in os.walk("data/Radvess"):
#     for file in files:

#         if file.endswith(".wav"):

#             emotion_code = file.split("-")[2]

#             emotion = emotion_map[emotion_code]

#             emotion_counter[emotion] += 1

# print(emotion_counter)

# import os

# print(os.listdir("data/AudioWAV")[:10])

# import os
# from collections import Counter

# emotion_map = {
#     "ANG": "angry",
#     "DIS": "disgust",
#     "FEA": "fear",
#     "HAP": "happy",
#     "NEU": "neutral",
#     "SAD": "sad"
# }

# counter = Counter()

# for file in os.listdir("data/AudioWAV"):

#     if file.endswith(".wav"):

#         code = file.split("_")[2]

#         emotion = emotion_map[code]

#         counter[emotion] += 1

# print(counter)

import os

print(os.listdir("data/TESS Toronto emotional speech set data")[:10])


import os

folder = r"data/TESS Toronto emotional speech set data/OAF_angry"

print(os.listdir(folder)[:10])