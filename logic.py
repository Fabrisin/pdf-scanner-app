import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# === CONFIGURATION ===
IMAGE_PATH = "DewarpBook.jpg"
DISPLAY_WIDTH = 500
NUM_POINTS = 10
GRID_ROWS = 20

# === LOAD IMAGE ===
src_img = cv2.imread(IMAGE_PATH)
if src_img is None:
    raise ValueError("Image not found.")
original_h, original_w = src_img.shape[:2]
scale = DISPLAY_WIDTH / original_w
resized_img = cv2.resize(src_img, (DISPLAY_WIDTH, int(original_h * scale)))

# === FUNCTIONS ===
def refine_bottom_points_from_draw(image, xs):
    global rect_pts, drawing
    print("🖱 Draw a box around the bottom of the page (click and drag)")

    clone = image.copy()
    rect_pts = []
    drawing = False

    def draw_rect(event, x, y, flags, param):
        global rect_pts, drawing
        if event == cv2.EVENT_LBUTTONDOWN:
            rect_pts = [(x, y)]
            drawing = True
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            temp = image.copy()
            cv2.rectangle(temp, rect_pts[0], (x, y), (0, 255, 255), 1)
            cv2.imshow("Draw Bottom Area", temp)
        elif event == cv2.EVENT_LBUTTONUP:
            rect_pts.append((x, y))
            drawing = False
            cv2.destroyWindow("Draw Bottom Area")

    cv2.namedWindow("Draw Bottom Area")
    cv2.setMouseCallback("Draw Bottom Area", draw_rect)
    cv2.imshow("Draw Bottom Area", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(rect_pts) != 2:
        raise ValueError("Invalid region selected.")

    (x1, y1), (x2, y2) = rect_pts
    roi = image[min(y1, y2):max(y1, y2), min(x1, x2):max(x1, x2)]
    offset_y = min(y1, y2)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    bottom_pts = []
    for x in xs:
        rel_x = x - min(x1, x2)
        if rel_x < 0 or rel_x >= edges.shape[1]:
            bottom_pts.append((x, offset_y + roi.shape[0] - 1))
            continue

        col = edges[:, rel_x]
        y_indices = np.where(col > 0)[0]
        if len(y_indices) > 0:
            bottom_y = int(np.percentile(y_indices, 95))
        else:
            bottom_y = roi.shape[0] - 1

        bottom_pts.append((x, offset_y + bottom_y))

    return bottom_pts

def auto_generate_top_points(image, num_points=10):
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    vertical_sum = np.sum(edges, axis=0)
    smoothed = np.convolve(vertical_sum, np.ones(20)/20, mode='same')
    threshold = np.max(smoothed) * 0.3
    mask = smoothed > threshold
    candidate_cols = np.where(mask)[0]

    if len(candidate_cols) < num_points:
        step = w // (num_points + 1)
        xs = [step * (i + 1) for i in range(num_points)]
    else:
        left, right = candidate_cols[0], candidate_cols[-1]
        xs = np.linspace(left, right, num_points).astype(int).tolist()

    top_pts = []
    for x in xs:
        col = edges[:, x]
        y_indices = np.where(col > 0)[0]
        top_y = y_indices[0] if len(y_indices) > 0 else int(h * 0.15)
        top_pts.append((x, top_y))

    return top_pts, xs

def label_point(img, x, y, i, color):
    cv2.circle(img, (x, y), 5, color, -1)
    cv2.putText(img, str(i + 1), (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

def draw_lines(img, top, bottom):
    for i in range(NUM_POINTS):
        cv2.line(img, top[i], bottom[i], (255, 0, 255), 1)
        label_point(img, *top[i], i, (0, 255, 0))
        label_point(img, *bottom[i], i, (0, 0, 255))
    for i in range(NUM_POINTS - 1):
        cv2.line(img, top[i], top[i + 1], (0, 150, 0), 1)
        cv2.line(img, bottom[i], bottom[i + 1], (0, 0, 150), 1)

def adjust_points(event, x, y, flags, param):
    global selected_idx
    if event == cv2.EVENT_LBUTTONDOWN:
        for i, pt in enumerate(adjust_pts):
            if abs(x - pt[0]) < 10 and abs(y - pt[1]) < 10:
                selected_idx = i
                break
    elif event == cv2.EVENT_MOUSEMOVE and selected_idx is not None:
        adjust_pts[selected_idx] = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        selected_idx = None

# === AUTO DETECT + ADJUST ===
top_pts, xs = auto_generate_top_points(resized_img, NUM_POINTS)
bottom_pts = refine_bottom_points_from_draw(resized_img, xs)
adjust_pts = top_pts + bottom_pts
selected_idx = None

cv2.namedWindow("Adjust Points (drag, ENTER to confirm)")
cv2.setMouseCallback("Adjust Points (drag, ENTER to confirm)", adjust_points)

while True:
    view = resized_img.copy()
    draw_lines(view, adjust_pts[:NUM_POINTS], adjust_pts[NUM_POINTS:])
    cv2.imshow("Adjust Points (drag, ENTER to confirm)", view)
    key = cv2.waitKey(1) & 0xFF
    if key == 13:
        break
cv2.destroyWindow("Adjust Points (drag, ENTER to confirm)")

# === SCALE POINTS BACK TO ORIGINAL IMAGE SIZE ===
top_pts_scaled = [(int(x / scale), int(y / scale)) for x, y in adjust_pts[:NUM_POINTS]]
bottom_pts_scaled = [(int(x / scale), int(y / scale)) for x, y in adjust_pts[NUM_POINTS:]]

# === DETERMINE OUTPUT CANVAS SIZE ===
(x0, y0), (x1, y1) = top_pts_scaled[0], top_pts_scaled[-1]
page_width = int(math.hypot(x1 - x0, y1 - y0))
page_height = int(np.mean([b[1] - t[1] for t, b in zip(top_pts_scaled, bottom_pts_scaled)]))

# grid cell size
total_strips = NUM_POINTS - 1
strip_w = float(page_width) / total_strips
cell_h = float(page_height) / GRID_ROWS

# create blank output canvas
dst_img = np.zeros((page_height, page_width, 3), dtype=src_img.dtype)

# === PIECEWISE HOMOGRAPHY WARP ===
for i in range(total_strips):
    for j in range(GRID_ROWS):
        t0 = np.array(top_pts_scaled[i], dtype=np.float32)
        t1 = np.array(top_pts_scaled[i + 1], dtype=np.float32)
        b0 = np.array(bottom_pts_scaled[i], dtype=np.float32)
        b1 = np.array(bottom_pts_scaled[i + 1], dtype=np.float32)
        alpha0 = j / GRID_ROWS
        alpha1 = (j + 1) / GRID_ROWS

        src_quad = np.array([
            (1 - alpha0) * t0 + alpha0 * b0,
            (1 - alpha0) * t1 + alpha0 * b1,
            (1 - alpha1) * t1 + alpha1 * b1,
            (1 - alpha1) * t0 + alpha1 * b0
        ], dtype=np.float32)

        dst_quad = np.array([
            [i * strip_w, j * cell_h],
            [(i + 1) * strip_w, j * cell_h],
            [(i + 1) * strip_w, (j + 1) * cell_h],
            [i * strip_w, (j + 1) * cell_h]
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_quad, dst_quad)
        warped = cv2.warpPerspective(src_img, H, (page_width, page_height),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_REPLICATE)

        mask = np.zeros((page_height, page_width), dtype=np.uint8)
        cv2.fillConvexPoly(mask, dst_quad.astype(np.int32), 255)
        dst_img[mask == 255] = warped[mask == 255]

# === CONTRAST ENHANCEMENT FUNCTION (SOFT)
def enhance_contrast_soft(image, clip_limit=2.0, tile_grid_size=(8, 8), gamma=1.2):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    gamma_inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** gamma_inv * 255 for i in range(256)]).astype("uint8")
    img_gamma = cv2.LUT(img_clahe, table)

    return img_gamma

# === ENHANCE CONTRAST
enhanced = enhance_contrast_soft(dst_img)

# === DISPLAY & SAVE
cv2.imshow("Final Dewarped Result (Enhanced)", enhanced)
cv2.imwrite("outputfinal.jpg", enhanced)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("✅ Saved: output_final_piecewise_softcontrast.jpg")