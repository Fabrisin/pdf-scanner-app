import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np

class CornerCorrectionCanvas(tk.Canvas):
    def __init__(self, master, image, initial_pts, on_done, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.image = image
        self.orig_img = image.copy()
        self.display_img = None
        self.tk_img = None

        self.pts = np.array(initial_pts, dtype=np.float32)
        self.on_done = on_done
        self.radius = 10
        self.drag_idx = None

        self.bind("<ButtonPress-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        self.draw()

    def draw(self):
        img = self.orig_img.copy()
        for pt in self.pts:
            cv2.circle(img, tuple(pt.astype(int)), self.radius, (0, 0, 255), -1)
        cv2.polylines(img, [self.pts.astype(int)], isClosed=True, color=(255, 0, 0), thickness=2)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.display_img = Image.fromarray(rgb)
        self.tk_img = ImageTk.PhotoImage(self.display_img)
        self.create_image(0, 0, anchor="nw", image=self.tk_img)

    def on_click(self, event):
        for i, pt in enumerate(self.pts):
            if np.linalg.norm(np.array([event.x, event.y]) - pt) < self.radius:
                self.drag_idx = i
                break

    def on_drag(self, event):
        if self.drag_idx is not None:
            self.pts[self.drag_idx] = [event.x, event.y]
            self.draw()

    def on_release(self, event):
        self.drag_idx = None

    def get_points(self):
        return self.pts

    def confirm(self):
        self.on_done(self.pts)
