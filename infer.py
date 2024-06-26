

import argparse
from ultralytics import YOLO

parser = argparse.ArgumentParser(description="Yolov8 inference script")
parser.add_argument(
    "--model",
    type=str,
    default="/Users/thanakrit/Taco_3/litter-detection/runs/detect/train/yolov8s_100epochs/weights/best.pt",
    help="path to yolo weights"
    )
parser.add_argument(
    "--source",
    type=str,
    default="/Users/thanakrit/Taco_3/litter-detection/assets/test.jpg",
    help="path to data to infer on"
)
parser.add_argument(
    "--save",
    action="store_true",
    help="save predictions"
)

if __name__=="__main__":
    args=parser.parse_args()
    
    model = YOLO(args.model)
    results = model.predict(source=args.source, save=args.save)

