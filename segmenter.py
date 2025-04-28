import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms

class DocumentScanner:
    def __init__(self, model_path: str, input_size=(512, 512)):
        # initialize scanner with model and preprocessing settings
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.4611, 0.4359, 0.3905), std=(0.2193, 0.2150, 0.2109))
        ])

    def _load_model(self, model_path):
        # Load DeepLabV3 model and custom weights 
        model = deeplabv3_resnet50(weights=None, aux_loss=True)
        model.classifier[4] = nn.LazyConv2d(2, 1)            # Customize output for binary segmentation
        model.aux_classifier[4] = nn.LazyConv2d(2, 1)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model.to(self.device)

    def _order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image, pts):
        # perform perspective transform
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        # Compute width and height of the new image
        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))
        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        # Destination rectangle for warp
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        # Warp perspective
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def scan_with_points(self, image_path):
        # Run segmentation and return original image, binary mask, and 4 point contour
        orig = cv2.imread(image_path)
        resized = cv2.resize(orig, self.input_size)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Transform image for model input
        tensor_img = self.transform(rgb).unsqueeze(0).to(self.device)

        # inference
        with torch.no_grad():
            out = self.model(tensor_img)["out"]
            pred = torch.argmax(out.squeeze(), dim=0).cpu().numpy()

        # resize predicted mask to og size
        mask = (pred * 255).astype(np.uint8)
        mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]))

        # clean up mask
        mask_clean = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask_clean = cv2.dilate(mask_clean, np.ones((5, 5), np.uint8), iterations=1)

        # combine large valid contours into a binary main one
        combined = np.zeros_like(mask_clean)
        contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                cv2.drawContours(combined, [cnt], -1, 255, -1)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("No contours found in mask.")

        # find largest countour an try to detet 4 points
        contour = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.01 * peri, True)

        # If 4 corners are found, use them; otherwise, use minimum area rectangle
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype(np.float32)
        else:
            rect = cv2.minAreaRect(cv2.convexHull(contour))
            box = cv2.boxPoints(rect)
            pts = box.astype(np.float32)

        return orig, mask, pts
