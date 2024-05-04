import argparse
import cv2
import supervision as sv

from src.segmentors import SamSegmentor
from src.filters import EllipseFilter
from src.utils import plot_ellipses_on_image


def main(input_path):
    # Initialize the segmentor and ellipse filter
    segmentor = SamSegmentor()
    ellipseFilter = EllipseFilter()

    # Load the input image
    image = cv2.imread(input_path)

    # Generate masks using the segmentor
    results = segmentor.mask_generator.generate(image)
    detections = sv.Detections.from_sam(sam_result=results)
    annotated_image = sv.MaskAnnotator().annotate(scene=image.copy(), detections=detections)

    # Filter the ellipses
    ellipse_df = ellipseFilter.filter_ellipses(detections)

    # Print the number of masks and ellipses
    print(f"{len(detections.mask)} masks")
    print(f"{len(ellipse_df)} ellipses")

    # Save ellipses_df to a CSV file
    ellipse_df.to_csv(f"{input_path.split('.')[0]}_ellipses.csv")

    # Output the annotated image and image with overlaid ellipses
    cv2.imwrite(f"{input_path.split('.')[0]}_segmented.png", annotated_image)
    plot_ellipses_on_image(image, ellipse_df, f"{input_path.split('.')[0]}_ellipses.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('input_path', type=str, help='Path to input image')

    args = parser.parse_args()

    main(args.input_path)
