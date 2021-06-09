# # Specify device
import os

# Specify device
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Import all necessary libraries.
import numpy as np
import sys
import cv2

# NomeroffNet path
NOMEROFF_NET_DIR = os.path.abspath('D:\\PycharmProjects\\RecognitionSystemFull\\nomeroff')

sys.path.append(NOMEROFF_NET_DIR)

# Import license plate recognition tools.
from nomeroff.NomeroffNet.YoloV5Detector import Detector
detector = Detector()
detector.load()

from nomeroff.NomeroffNet.BBoxNpPoints import NpPointsCraft, getCvZoneRGB, convertCvZonesRGBtoBGR, reshapePoints
npPointsCraft = NpPointsCraft()
npPointsCraft.load()

from nomeroff.NomeroffNet.OptionsDetector import OptionsDetector
from nomeroff.NomeroffNet.TextDetector import TextDetector

from nomeroff.NomeroffNet import TextDetector
from nomeroff.NomeroffNet import textPostprocessing

import telebot
from io import BytesIO

# load models
optionsDetector = OptionsDetector()
optionsDetector.load("latest")

textDetector = TextDetector.get_static_module("ru")()
textDetector.load("latest")


print("Bot start")
bot = telebot.TeleBot('1836160558:AAGOo4OkFcdsRQMnnW49h9tz1o-2XxcYkzw')

@bot.message_handler(content_types=['photo'])
def handle(message):
    fileId = message.photo[-1].file_id
    file = bot.get_file(fileId)
    down_file = bot.download_file(file.file_path)
    with open("image.jpg", "wb") as f:
         f.write(down_file)
    img = cv2.imread("image.jpg")
    bot.send_message(message.chat.id, "Начало расппознавания")
    targetBoxes = detector.detect_bbox(img)
    all_points = npPointsCraft.detect(img, targetBoxes, [5, 2, 0])

    # cut zones

    zones = convertCvZonesRGBtoBGR([getCvZoneRGB(img, reshapePoints(rect, 1)) for rect in all_points])

    # predict zones attributes
    regionIds, stateIds, countLines = optionsDetector.predict(zones)
    regionNames = optionsDetector.getRegionLabels(regionIds)

    # find text with postprocessing by standart
    textArr = textDetector.predict(zones)
    textArr = textPostprocessing(textArr, regionNames)
    img = BytesIO(open("image.jpg", "rb").read())
    if len(textArr) == 0:
        bot.send_photo(message.chat.id, img, "На фотографии не был найден номерной знак")
    else:
        bot.send_photo(message.chat.id, img, textArr)

bot.polling(none_stop = True)
