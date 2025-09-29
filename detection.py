def ship_detection(model=None, img =None):
    """
    Detect ships in an patch image using a YOLO model.

    Args:
        model: Pre-loaded YOLO model
        img: numpy  q

    Returns:
        detection_infors (list): A list containing bounding box coordinates and confidence scores.
            Each element is structured as:
                [xyxy, conf]
            where:
                - xyxy: numpy array (n, 4), coordinates of bounding boxes (x1, y1, x2, y2).
                - conf: numpy array (n,), confidence scores for each box.
    """
    results = model(img)
    detection_infors = []
    for r in results:
        boxes = r.boxes
        xyxy = boxes.xyxy.cpu().numpy()  # (x1, y1, x2, y2)
        conf = boxes.conf.cpu().numpy()
        a_box = [xyxy, conf]
        detection_infors.append(a_box)
    return detection_infors
    