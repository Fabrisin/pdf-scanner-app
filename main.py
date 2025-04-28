import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from segmenter import DocumentScanner
import os
import numpy as np
import matplotlib.pyplot as plt

class DewarpApp:
    def __init__(self, master):
        # Main window setu0p
        self.master = master
        self.master.title("🧾 Document Dewarping App")
        self.master.geometry("900x600")
        self.master.configure(bg="#1e1e1e")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton",
                        font=("Segoe UI", 11),
                        padding=10,
                        relief="flat",
                        background="#3b82f6",
                        foreground="white")
        style.map("TButton", background=[("active", "#2563eb")])

        button_frame = tk.Frame(master, bg="#1e1e1e")
        button_frame.pack(side="top", fill="x", pady=10)

        self.load_btn = ttk.Button(button_frame, text="📂 Load Image", command=self.load_image)
        self.load_btn.pack(side="left", padx=10)

        self.ml_btn = ttk.Button(button_frame, text="🤖 ML-Based Scan", command=self.ml_based)
        self.ml_btn.pack(side="left", padx=10)

        self.canvas_frame = tk.Frame(master, bg="#111827", bd=2, relief="ridge")
        self.canvas_frame.pack(fill="both", expand=True, padx=20, pady=10)
        self.canvas = tk.Canvas(self.canvas_frame, bg="#111827", highlightthickness=2, highlightbackground="#111827")
        self.canvas.pack(fill="both", expand=True)

        # dnd support
        try:
            from tkinterdnd2 import DND_FILES
            self.canvas.drop_target_register(DND_FILES)
            self.canvas.dnd_bind('<<Drop>>', self.on_file_drop)
            self.canvas.dnd_bind('<<DragEnter>>', self.on_drag_enter)
            self.canvas.dnd_bind('<<DragLeave>>', self.on_drag_leave)
        except:
            pass

        # image and model init
        self.tk_img = None
        self.original_output = None
        self.filtered_output = None
        self.scanner = DocumentScanner("document_segmenter.pth")  # Load ML model

    def load_image(self, path=None):
        # load the image (via DnD or dialog)
        if not path:
            path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
        if not path:
            return
        self.ml_image_path = path
        self.display_image(cv2.imread(path))

    def ml_based(self):
        # run the document segmenter and corner adjustment
        if not hasattr(self, "ml_image_path"):
            messagebox.showwarning("Warning", "Please load an image first.")
            return
        try:
            image, mask, pts = self.scanner.scan_with_points(self.ml_image_path)
            self._interactive_adjust_in_gui(image, mask, pts)
        except Exception as e:
            messagebox.showerror("ML Error", str(e))

    def _interactive_adjust_in_gui(self, image, mask, pts):
        # display interactive plot 
        fig, ax = plt.subplots()
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        points_plot, = ax.plot(pts[:, 0], pts[:, 1], 'ro', markersize=8, picker=5)
        polygon_plot, = ax.plot(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]), 'b-')

        moving_idx = [None]

        def on_pick(event):
            for i, pt in enumerate(pts):
                if np.linalg.norm(pt - np.array([event.mouseevent.xdata, event.mouseevent.ydata])) < 10:
                    moving_idx[0] = i
                    break

        def on_release(event):
            moving_idx[0] = None

        def on_motion(event):
            if moving_idx[0] is not None and event.xdata and event.ydata:
                pts[moving_idx[0]] = [event.xdata, event.ydata]
                points_plot.set_data(pts[:, 0], pts[:, 1])
                polygon_plot.set_data(np.append(pts[:, 0], pts[0, 0]), np.append(pts[:, 1], pts[0, 1]))
                fig.canvas.draw_idle()

        # bindings for interactive adjustments
        fig.canvas.mpl_connect('pick_event', on_pick)
        fig.canvas.mpl_connect('motion_notify_event', on_motion)
        fig.canvas.mpl_connect('button_release_event', on_release)
        plt.title("Drag red points to adjust corners. Close the window when done.")
        plt.show()

        # apply perspective transform
        result = self.scanner._four_point_transform(image, pts)
        self.original_output = result.copy()
        self.filtered_output = result.copy()
        self.setup_filter_ui()
        self.update_filter_canvas()

    def display_image(self, image):
        # display image function
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_w, img_h = img_pil.size

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))
        img_pil = img_pil.resize(new_size, resample=Image.LANCZOS)

        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        x_offset = (canvas_w - new_size[0]) // 2
        y_offset = (canvas_h - new_size[1]) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img)

    def setup_filter_ui(self):
        # UI for filter selection
        if hasattr(self, 'filter_frame'):
            return

        self.filter_frame = tk.Frame(self.master, bg="#1e1e1e")
        self.filter_frame.pack(side="bottom", fill="x", pady=5)

        filters = ["None", "Soft Contrast", "Sharpen", "Grayscale", "Sepia", "Invert"]
        self.selected_filter = tk.StringVar(value="None")

        for f in filters:
            ttk.Radiobutton(self.filter_frame, text=f, value=f, variable=self.selected_filter,
                            command=self.apply_selected_filter).pack(side="left", padx=10)

        ttk.Button(self.filter_frame, text="💾 Save Result", command=self.save_filtered_image).pack(side="right", padx=20)

    def apply_selected_filter(self):
        # apply filters to the raw warped image
        choice = self.selected_filter.get()
        base = self.original_output

        if choice == "None":
            filtered = base.copy()
        elif choice == "Soft Contrast":
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, 25, 10)
            filtered = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        elif choice == "Sharpen":
            sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            filtered = cv2.filter2D(base, -1, sharpen_kernel)
        elif choice == "Grayscale":
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
            filtered = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif choice == "Sepia":
            sepia = np.array([[0.272, 0.534, 0.131],
                              [0.349, 0.686, 0.168],
                              [0.393, 0.769, 0.189]])
            filtered = cv2.transform(base, sepia)
            filtered = np.clip(filtered, 0, 255).astype(np.uint8)
        elif choice == "Invert":
            filtered = cv2.bitwise_not(base)
        else:
            filtered = base

        self.filtered_output = filtered
        self.update_filter_canvas()

    def update_filter_canvas(self):
        # refresh the window after applying a filter
        image = self.filtered_output
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        canvas_w = self.canvas.winfo_width() or 900
        canvas_h = self.canvas.winfo_height() or 600
        img_w, img_h = img_pil.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_size = (int(img_w * scale), int(img_h * scale))

        img_pil = img_pil.resize(new_size, resample=Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(img_pil)

        self.canvas.delete("all")
        x_offset = (canvas_w - new_size[0]) // 2
        y_offset = (canvas_h - new_size[1]) // 2
        self.canvas.create_image(x_offset, y_offset, anchor="nw", image=self.tk_img)

    def on_file_drop(self, event):
        # Drag and drop functionality
        path = event.data.strip('{}') if '{' in event.data else event.data
        if os.path.isfile(path):
            self.load_image(path)

    def on_drag_enter(self, event):
        self.canvas.config(highlightbackground="#3b82f6")

    def on_drag_leave(self, event):
        self.canvas.config(highlightbackground="#111827")

    def save_filtered_image(self):
        # Save final image
        cv2.imwrite("dewarped_output.jpg", self.filtered_output)
        messagebox.showinfo("Done", "✅ Dewarped image saved as 'dewarped_output.jpg'")

if __name__ == "__main__":
    # App initialization
    try:
        from tkinterdnd2 import DND_FILES, TkinterDnD
        root = TkinterDnD.Tk()
    except ImportError:
        TkinterDnD = None
        root = tk.Tk()
    app = DewarpApp(root)
    root.mainloop()
