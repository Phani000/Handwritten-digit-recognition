import pygame
from pygame.locals import *
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model

WINDOWSIZEX = 640
WINDOWSIZEY = 480

BOUNDRYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

IMAGESAVE = False

try:
    MODEL = load_model("model/hwdr_model.keras")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit()

LABELS = {0: "Zero",
          1: "One",
          2: "Two",
          3: "Three",
          4: "Four",
          5: "Five",
          6: "Six",
          7: "Seven",
          8: "Eight",
          9: "Nine"
          }

# Initialize pygame
pygame.init()

FONT = pygame.font.Font(None, 18)  # Using a system font as a safer default
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))

pygame.display.set_caption("Digital Board")

iswriting = False

number_xcord = []
number_ycord = []
image_count = 1
PREDICT = True

DISPLAYSURF.fill(BLACK)

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()

        if event.type == MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)

            number_xcord.append(xcord)
            number_ycord.append(ycord)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
            number_xcord = []
            number_ycord = []
            DISPLAYSURF.fill(BLACK) # Clear the screen on new drawing

        if event.type == MOUSEBUTTONUP and iswriting:
            iswriting = False
            if number_xcord and number_ycord:
                rect_min_x, rect_max_x = min(number_xcord) - BOUNDRYINC, max(number_xcord) + BOUNDRYINC
                rect_min_y, rect_max_y = min(number_ycord) - BOUNDRYINC, max(number_ycord) + BOUNDRYINC

                if rect_min_x < 0:
                    rect_min_x = 0
                if rect_max_x > WINDOWSIZEX:
                    rect_max_x = WINDOWSIZEX
                if rect_min_y < 0:
                    rect_min_y = 0
                if rect_max_y > WINDOWSIZEY:
                    rect_max_y = WINDOWSIZEY

                drawn_area = DISPLAYSURF.subsurface((rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y))
                img_arr = pygame.surfarray.array3d(drawn_area).astype(np.float32)
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

                if IMAGESAVE:
                    cv2.imwrite(f"image_{image_count}.png", img_arr)
                    image_count += 1

                if PREDICT:
                    resized_img = cv2.resize(img_arr, (28, 28))
                    normalized_img = resized_img / 255.0
                    reshaped_img = normalized_img.reshape(1, 28, 28, 1)
                    prediction = MODEL.predict(reshaped_img)
                    predicted_label = str(LABELS[np.argmax(prediction)])

                    textSurface = FONT.render(predicted_label, True, RED, WHITE)
                    textRecObj = textSurface.get_rect()
                    textRecObj.topleft = (rect_min_x, rect_max_y + 5) # Adjust text position

                    DISPLAYSURF.blit(textSurface, textRecObj)

                number_xcord = []
                number_ycord = []

        if event.type == KEYDOWN:
            if event.unicode == 'n':
                DISPLAYSURF.fill(BLACK)

    pygame.display.update()
