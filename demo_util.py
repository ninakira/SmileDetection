import cv2
import numpy as np

IMG_WIDTH = 416
IMG_HEIGHT = 416

CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4

# Default colors
COLOR_BLUE = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 255, 255)


def get_pred_color(pred):
    return COLOR_GREEN if pred >= 0.5 else COLOR_WHITE


def refined_box(left, top, width, height):
    right = left + width
    bottom = top + height

    original_vert_height = bottom - top
    top = int(top + original_vert_height * 0.15)
    bottom = int(bottom - original_vert_height * 0.05)

    margin = ((bottom - top) - (right - left)) // 2
    left = left - margin if (bottom - top - right + left) % 2 == 0 else left - margin - 1

    right = right + margin

    return left, top, right, bottom


# Draw the predicted bounding box
def draw_predict(frame, left, top, right, bottom, conf=None):
    # Draw a bounding box.
    cv2.rectangle(frame, (left, top), (right, bottom), COLOR_YELLOW, 2)

    if conf is not None:
        text = '{:.2f}'.format(conf)

        # Display the label at the top of the bounding box
        label_size, base_line = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        top = max(top, label_size[1])
        cv2.putText(frame, text, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    get_pred_color(conf), 1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))
