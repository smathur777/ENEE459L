import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw
import warnings
from scipy.ndimage import convolve, gaussian_filter

MODEL_PATH = "version-RFB-320.onnx"

warnings.filterwarnings("ignore", category=UserWarning)

sess_options = ort.SessionOptions()
sess_options.log_severity_level = 3  

# 3. Use only available providers to prevent the provider fallback warning
desired_providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
available_providers = [p for p in desired_providers if p in ort.get_available_providers()]

# face detection
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :4]
    indexes = np.argsort(scores)[-candidate_size:]
    picked = []

    while indexes.size > 0:
        current = indexes[-1]
        picked.append(current)
        if len(picked) == top_k or indexes.size == 1:
            break

        indexes = indexes[:-1]
        remaining_boxes = boxes[indexes]
        current_box = boxes[current]
        left_top = np.maximum(current_box[:2], remaining_boxes[:, :2])
        right_bottom = np.minimum(current_box[2:], remaining_boxes[:, 2:])
        intersection_size = np.maximum(0, right_bottom - left_top)
        intersection = intersection_size[:, 0] * intersection_size[:, 1]

        current_size = np.maximum(0, current_box[2:] - current_box[:2])
        remaining_sizes = np.maximum(0, remaining_boxes[:, 2:] - remaining_boxes[:, :2])
        current_area = current_size[0] * current_size[1]
        remaining_areas = remaining_sizes[:, 0] * remaining_sizes[:, 1]
        union = current_area + remaining_areas - intersection
        iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float),
                        where=union > 0)
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked]

def detect_faces(image_path, output_path="output_detected.png", conf_threshold=0.7):
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    width, height = image.size

    input_tensor = np.asarray(image.resize((320, 240)), dtype=np.float32)
    input_tensor = (input_tensor - 127.0) / 128.0
    input_tensor = np.expand_dims(input_tensor.transpose(2, 0, 1), axis=0)

    session = ort.InferenceSession(MODEL_PATH, sess_options=sess_options,
                                   providers=available_providers)
    confidences, boxes = session.run(None, {session.get_inputs()[0].name: input_tensor})
    scores = confidences[0, :, 1]
    boxes = boxes[0]
    mask = scores >= conf_threshold
    box_scores = np.concatenate((boxes[mask], scores[mask, None]), axis=1)
    detections = hard_nms(box_scores, iou_threshold=0.3)

    draw = ImageDraw.Draw(image)
    for detection in detections:
        box = (detection[:4] * np.array([width, height, width, height])).astype(int)
        draw.rectangle(tuple(box), outline="lime", width=3)
        draw.text((int(box[0]) + 5, int(box[1]) - 15),
                  f"Face: {detection[4]:.2f}", fill="lime")
    image.save(output_path)

# Edge detection
def non_maximum_suppression(magnitude, angle):
    height, width = magnitude.shape
    result = np.zeros((height, width), dtype=np.float32)
    angle = np.rad2deg(angle) % 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            direction = angle[i, j]
            if direction < 22.5 or direction >= 157.5:
                first, second = magnitude[i, j - 1], magnitude[i, j + 1]
            elif direction < 67.5:
                first, second = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif direction < 112.5:
                first, second = magnitude[i - 1, j], magnitude[i + 1, j]
            else:
                first, second = magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]

            if magnitude[i, j] >= first and magnitude[i, j] >= second:
                result[i, j] = magnitude[i, j]

    return result

def hysteresis_thresholding(img, low_thresh, high_thresh):
    strong, weak = 255, 50
    result = np.zeros(img.shape, dtype=np.uint8)
    result[img >= high_thresh] = strong
    weak_i, weak_j = np.where((img >= low_thresh) & (img < high_thresh))
    result[weak_i, weak_j] = weak

    height, width = img.shape
    for i, j in zip(weak_i, weak_j):
        neighborhood = result[max(0, i - 1):min(height, i + 2),
                              max(0, j - 1):min(width, j + 2)]
        result[i, j] = strong if np.any(neighborhood == strong) else 0

    return result

def canny_scipy(image_path, output_path="edges_scipy.png", low=20, high=60):
    with Image.open(image_path) as source:
        image = np.asarray(source.convert("L"), dtype=np.float32)
    smoothed = gaussian_filter(image, sigma=1.4)

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = convolve(smoothed, kx)
    gy = convolve(smoothed, ky)
    magnitude = np.hypot(gx, gy)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255
    angle = np.arctan2(gy, gx)

    thinned = non_maximum_suppression(magnitude, angle)
    edges = hysteresis_thresholding(thinned, low, high)
    Image.fromarray(edges).save(output_path)

if __name__ == "__main__":
    detect_faces("face.jpg")
    canny_scipy("face.jpg")
