import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def draw_box_with_label(frame, box, label, color=(0, 255, 0), thickness=2):
    """
    Draw a bounding box with a label.
    box: (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.5, 1)
    label_bg_tl = (x1, max(0, y1 - th - 6))
    label_bg_br = (x1 + tw + 6, max(0, y1 - 2))
    cv2.rectangle(frame, label_bg_tl, label_bg_br, (30, 30, 30), -1)
    cv2.putText(frame, label, (x1 + 3, label_bg_br[1] - 4),
                FONT, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
