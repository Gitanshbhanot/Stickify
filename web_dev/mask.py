import os
import cv2
import numpy as np
import mediapipe as mp
# store background images in a list

mp_selfie_segmentation = mp.solutions.selfie_segmentation
selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

bgrd_image = cv2.imread('./web_dev/bg_image.jpg')

def Mask(frame):
  # flip the frame to horizontal direction
  frame = cv2.flip(frame, 1)
  height , width, channel = frame.shape

  RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  # get the result
  results = selfie_segmentation.process(RGB)
  # extract segmented mask
  mask = results.segmentation_mask
  # show outputs
  # cv2.imshow("mask", mask)
  # cv2.imshow("Frame", frame)

  condition = np.stack(
    (results.segmentation_mask,) * 3, axis=-1) > 0.5
  # resize the background image to the same size of the original frame
  global bgrd_image
  bgrd_image = cv2.resize(bgrd_image, (width, height))
  # combine frame and background image using the condition
  output_image = np.where(condition, frame, bgrd_image)
  cv2.imwrite(filename='./static/remove_background.jpg', img=output_image)
  cv2.imshow("Output", output_image)
  # cv2.imshow("Frame", frame)