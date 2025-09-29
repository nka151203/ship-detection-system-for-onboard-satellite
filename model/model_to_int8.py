from ultralytics import YOLO
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def int8_model():
    model_path = os.path.join(BASE_DIR,"/8nqat_int8_openvino_model/")
    # Load model
    if os.path.exists(model_path) and os.path.isdir(model_path):
        model = YOLO(os.path.join(BASE_DIR,"8nqat.pt"))
        # Export OpenVINO INT8
        model.export(format="openvino", int8=True)

    int_8_model = YOLO(os.path.join(BASE_DIR,"8nqat_int8_openvino_model"))
    return int_8_model


