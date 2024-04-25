import argparse
import cv2
import supervision as sv

from src.segmentors import SamSegmentor


def main(input_file):
    image = cv2.imread(input_file)

    segmentor = SamSegmentor()
    results = segmentor.mask_generator.generate(image)
    detections = sv.Detections.from_sam(sam_result=results)
    annotated_image = sv.MaskAnnotator().annotate(scene=image.copy(), detections=detections)

    cv2.imwrite(f"{input_file.split('.')[0]}_segmented_output.png", annotated_image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_file', type=str, help='Path to input image')

    args = parser.parse_args()

    main(args.input_file)
