import cv2
import numpy as np

def apply_pattern_to_flag(flag_path, pattern_path, output_path="output_flag.jpg"):
    # Load images
    flag_img = cv2.imread(flag_path, cv2.IMREAD_COLOR)
    pattern_img = cv2.imread(pattern_path, cv2.IMREAD_COLOR)

    if flag_img is None:
        raise FileNotFoundError(f"Flag image not found at: {flag_path}")
    if pattern_img is None:
        raise FileNotFoundError(f"Pattern image not found at: {pattern_path}")

    # Resize pattern to match flag dimensions
    pattern_resized = cv2.resize(pattern_img, (flag_img.shape[1], flag_img.shape[0]))

    rows, cols = flag_img.shape[:2]

    # --- 1. Create a mask for the flag area (assuming white flag on white bg) ---
    gray = cv2.cvtColor(flag_img, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)  # Invert: flag is white, bg is white

    # Optional: clean up mask
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # --- 2. Simulate flag waving (as before) ---
    map_x, map_y = np.meshgrid(np.arange(cols), np.arange(rows))
    amplitude = 15
    frequency = 2
    phase = np.pi / 3
    offset_x = amplitude * np.sin(2 * np.pi * map_y / rows * frequency + phase)
    offset_y = amplitude * 0.2 * np.sin(2 * np.pi * map_x / cols * frequency + phase)
    map_x_warped = (map_x + offset_x).astype(np.float32)
    map_y_warped = (map_y + offset_y).astype(np.float32)
    warped_pattern = cv2.remap(pattern_resized, map_x_warped, map_y_warped, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    # --- 3. Blend only where the flag is ---
    mask_3ch = cv2.merge([mask, mask, mask])
    mask_norm = mask_3ch / 255.0

    output_img = flag_img.copy()
    output_img = (warped_pattern * mask_norm + flag_img * (1 - mask_norm)).astype(np.uint8)

    # --- 4. Save the result ---
    cv2.imwrite(output_path, output_img)
    print(f"Output saved to {output_path}")

# Example usage:
if __name__ == "__main__":
    apply_pattern_to_flag(
        flag_path="/Users/naresh/Downloads/flag-naresh/whiteflag.jpg",
        pattern_path="/Users/naresh/Downloads/flag-naresh/americaflag.jpg",
        output_path="Output.jpg"
    )
if __name__ == "__main__":
    output_img = cv2.imread("Output.jpg")
    cv2.imshow('Output Waving Flag', output_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()