# -*- coding: utf-8 -*-
"""
Fruit Caliper – Aforo (v4.1 con ROI, zoom/pan y auto-calibración de moneda)
Requisitos:
  pip install opencv-python numpy pillow pandas

Uso rápido:
  1) Archivo → Cargar fotos (Ctrl+O)
  2) 🪙 Auto-calibrar (o 📏 2 clics en diámetro de la moneda)
  3) 🟩 ROI (arrastrar) para limitar a la copa
  4) 👁️ Vista previa y ajusta parámetros (Min/Max mm, Color %, Hue σ, Edge ring %, Hough p2)
  5) Rellena Nº árboles y Árboles muestreados → ▶️ Procesar todas
"""

import os, math, cv2, numpy as np, pandas as pd
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

APP_TITLE = "Fruit Caliper – Aforo (v4.1)"
COIN_DIAMETER_MM = 23.25  # 1€ en mm

# ---------------- Utilidades ----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def distance(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])

def preprocess(bgr, blur):
    """Realce suave + blur opcional."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l,a,b = cv2.split(lab)
    l2 = cv2.createCLAHE(2.0,(8,8)).apply(l)
    enh = cv2.cvtColor(cv2.merge([l2,a,b]), cv2.COLOR_LAB2BGR)
    if blur>0: enh = cv2.GaussianBlur(enh, (blur|1, blur|1), 0)
    return enh

def make_roi_mask(shape, rect):
    """rect = (x1,y1,x2,y2) en coords de imagen base."""
    if rect is None: return None
    h,w = shape[:2]
    x1,y1,x2,y2 = [int(round(v)) for v in rect]
    x1,x2 = max(0,min(x1,w-1)), max(0,min(x2,w-1))
    y1,y2 = max(0,min(y1,h-1)), max(0,min(y2,h-1))
    if x2<=x1 or y2<=y1: return None
    m = np.zeros((h,w), np.uint8); m[y1:y2, x1:x2] = 255
    return m

def ring_mask(img, x, y, r, inner=0.80, outer=1.20):
    """Return a binary mask for an annulus centered at (x, y).

    The previous implementation expected a shape tuple but the callers
    provided the full image array. Using the array directly caused
    ``TypeError: only integer scalar arrays can be converted`` when
    creating the mask. Now the function derives the shape from the image
    itself, making the call sites consistent.
    """
    h, w = img.shape[:2]
    rr = np.zeros((h, w), np.uint8)
    cv2.circle(rr, (x, y), int(max(1, r * outer)), 255, -1)
    cv2.circle(rr, (x, y), int(max(1, r * inner)), 0, -1)
    return rr

# ------------- Auto moneda -------------------
def auto_detect_coin(bgr):
    """Devuelve (x,y,r, gold_ratio, edge_score) o None."""
    work = preprocess(bgr, blur=3)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    h,w = gray.shape[:2]
    min_r = max(8, int(0.01*min(h,w)))
    max_r = int(0.12*min(h,w))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r,
                               param1=160, param2=28, minRadius=min_r, maxRadius=max_r)
    if circles is None: return None
    circles = np.around(circles[0,:]).astype(int)  # <- sin uint16
    edges = cv2.Canny(gray, 80, 180)
    best, score_best = None, -1.0
    for (x,y,r) in circles:
        if x<r or y<r or x+r>=w or y+r>=h: continue
        ring = ring_mask(gray, x,y,r, 0.65, 1.05)
        ring_count = max(1, cv2.countNonZero(ring))
        gold = cv2.inRange(hsv,(15,60,60),(40,255,255))
        gold_ratio = cv2.countNonZero(cv2.bitwise_and(gold, gold, mask=ring))/ring_count
        edge_score = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ring)) / (2*math.pi*r + 1e-6)
        score = 0.6*edge_score + 0.4*gold_ratio
        if score>score_best:
            score_best, best = score, (int(x),int(y),int(r), float(gold_ratio), float(edge_score))
    if best and (best[3] >= 0.05 or best[4] >= 0.6):
        return best
    return None

# ------------- Detección de frutos -----------
def detect_candidates(bgr, px_per_mm, min_mm, max_mm, blur, hough_p2, roi_mask):
    """Candidatos por Hough (sin filtros de calidad todavía)."""
    work = preprocess(bgr, blur)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    min_r = max(5, int((min_mm/2.0)*px_per_mm))
    max_r = max(min_r+2, int((max_mm/2.0)*px_per_mm))
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(0.9*min_r),
        param1=140, param2=hough_p2, minRadius=min_r, maxRadius=max_r
    )
    cands = []
    h,w = gray.shape[:2]
    if circles is not None:
        for (x,y,r) in np.around(circles[0,:]).astype(int):  # <- sin uint16
            if x<r or y<r or x+r>=w or y+r>=h: continue
            if roi_mask is not None and roi_mask[y, x]==0: continue
            cands.append((x,y,r))
    return cands, work, (min_r, max_r)

def quality_filters(work, cands, color_ratio_min, hue_std_max, edge_ring_min, roi_mask=None):
    """Filtro por color (HSV), uniformidad de tono y borde en anillo."""
    hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(cv2.cvtColor(work, cv2.COLOR_BGR2GRAY), 70, 170)
    green = cv2.inRange(hsv, (20,30,35), (85,255,255))
    passed = []
    h,w = work.shape[:2]

    for (x,y,r) in cands:
        # máscara circular rellena (sin ogrid)
        circ = np.zeros((h,w), np.uint8)
        cv2.circle(circ,(x,y),r,255,-1)
        if roi_mask is not None: circ = cv2.bitwise_and(circ, roi_mask)

        area = max(1, cv2.countNonZero(circ))
        on_color = cv2.countNonZero(cv2.bitwise_and(green, green, mask=circ))
        color_ratio = on_color/area
        if color_ratio < color_ratio_min: 
            continue

        # Uniformidad de tono (σ de Hue)
        h_vals = hsv[:,:,0][circ.astype(bool)]
        if h_vals.size < 10: 
            continue
        hue_std = float(np.std(h_vals))
        if hue_std > hue_std_max:
            continue

        # Borde en anillo
        ring = ring_mask(work, x,y,r, 0.85, 1.15)
        if roi_mask is not None: ring = cv2.bitwise_and(ring, roi_mask)
        ring_count = max(1, cv2.countNonZero(ring))
        ring_edges = cv2.countNonZero(cv2.bitwise_and(edges, edges, mask=ring))
        ring_edge_frac = ring_edges / ring_count
        if ring_edge_frac < edge_ring_min:
            continue

        passed.append((x,y,r))

    return passed

def detect_by_contours(bgr, px_per_mm, min_mm, max_mm, blur, roi_mask):
    """Fallback robusto por contornos elípticos."""
    work = preprocess(bgr, blur)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    if roi_mask is not None:
        gray = cv2.bitwise_and(gray, gray, mask=roi_mask)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.dilate(edges, np.ones((3,3),np.uint8),1)
    contours,_ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = math.pi*((min_mm/2.0*px_per_mm)**2)*0.5
    max_area = math.pi*((max_mm/2.0*px_per_mm)**2)*1.6
    found = []
    h,w = work.shape[:2]
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area or len(c)<5: continue
        (x,y),(MA,ma),ang = cv2.fitEllipse(c)
        peri = cv2.arcLength(c, True)
        if peri==0: continue
        circ = 4*math.pi*area/(peri*peri)
        axis = min(MA,ma)/max(MA,ma)
        if circ>=0.6 and axis>=0.6:
            xr,yr = int(x),int(y)
            if roi_mask is not None and roi_mask[yr, xr]==0: continue
            r = int((MA+ma)/4.0)
            found.append((xr,yr,r))
    return found

# ------------- App -------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE); self.geometry("1220x880+60+40")

        # estado
        self.img_paths = []
        self.base_bgr = None        # imagen base
        self.display_bgr = None     # para overlay/preview
        self.zoom = 1.0
        self.px_per_mm = None

        # modos
        self.calibrating = False
        self.calib_pts = []
        self.roi_selecting = False
        self.roi_rect = None
        self.roi_start = None

        # Menú
        menubar = tk.Menu(self)
        fm = tk.Menu(menubar, tearoff=False)
        fm.add_command(label="Cargar fotos...    Ctrl+O", command=self.load_images)
        fm.add_separator(); fm.add_command(label="Salir", command=self.destroy)
        menubar.add_cascade(label="Archivo", menu=fm); self.config(menu=menubar)
        self.bind_all("<Control-o>", lambda e: self.load_images())

        # Barra principal
        self.ctrl = tk.Frame(self); self.ctrl.pack(fill="x")
        tk.Button(self.ctrl, text="➕ Cargar fotos", command=self.load_images).pack(side="left", padx=6, pady=6)
        tk.Button(self.ctrl, text="🪙 Auto-calibrar", command=self.auto_calibrate).pack(side="left", padx=6)
        tk.Button(self.ctrl, text="📏 Calibrar (2 clics)", command=self.start_calibration).pack(side="left", padx=6)
        tk.Button(self.ctrl, text="🟩 ROI (arrastrar)", command=self.start_roi).pack(side="left", padx=6)
        tk.Button(self.ctrl, text="✖ Quitar ROI", command=self.clear_roi).pack(side="left", padx=6)
        tk.Button(self.ctrl, text="👁️ Vista previa", command=self.preview_current).pack(side="left", padx=6)
        tk.Button(self.ctrl, text="▶️ Procesar todas", command=self.process_all).pack(side="left", padx=6)
        self.scale_lbl = tk.Label(self.ctrl, text="Escala: pendiente"); self.scale_lbl.pack(side="left", padx=12)

        # Zoom/pan
        tk.Button(self.ctrl, text="🔎 +", command=lambda: self.set_zoom(self.zoom*1.15)).pack(side="right", padx=4)
        tk.Button(self.ctrl, text="🔎 −", command=lambda: self.set_zoom(self.zoom/1.15)).pack(side="right", padx=4)

        # Parámetros de detección
        self.min_mm = tk.IntVar(value=50)
        self.max_mm = tk.IntVar(value=90)
        self.hough_p2 = tk.IntVar(value=24)
        self.blur = tk.IntVar(value=9)
        self.color_pct = tk.DoubleVar(value=0.12)   # 12%
        self.hue_std_max = tk.DoubleVar(value=12.0) # σ de Hue máx (0-180)
        self.edge_ring_min = tk.DoubleVar(value=0.22)

        def add_param(lbl, var, w=6):
            f = tk.Frame(self.ctrl); f.pack(side="left", padx=5)
            tk.Label(f, text=lbl).pack(); tk.Entry(f, width=w, textvariable=var).pack()
        add_param("Min mm", self.min_mm)
        add_param("Max mm", self.max_mm)
        add_param("Hough p2", self.hough_p2)
        add_param("Blur", self.blur)
        add_param("Color %", self.color_pct, 6)
        add_param("Hue σ máx", self.hue_std_max, 6)
        add_param("Edge ring %", self.edge_ring_min, 6)

        # Barra aforo
        self.aforo = tk.Frame(self); self.aforo.pack(fill="x", padx=8, pady=4)
        self.num_arboles = tk.IntVar(value=100)
        self.arboles_muestreados = tk.IntVar(value=1)
        self.densidad = tk.DoubleVar(value=0.95)
        self.factor_forma = tk.DoubleVar(value=0.85)
        self.modo_robusto = tk.BooleanVar(value=True)
        def add_field(lbl, var, w=8):
            f = tk.Frame(self.aforo); f.pack(side="left", padx=10)
            tk.Label(f, text=lbl).pack(); tk.Entry(f, width=w, textvariable=var).pack()
        add_field("Nº árboles parcela", self.num_arboles, 8)
        add_field("Árboles muestreados (fotos)", self.arboles_muestreados, 6)
        add_field("Densidad (g/cm³)", self.densidad, 6)
        add_field("Factor forma k", self.factor_forma, 6)
        tk.Checkbutton(self.aforo, text="🛟 Modo robusto (contornos)",
                       variable=self.modo_robusto).pack(side="left", padx=12)

        # Canvas
        self.canvas = tk.Canvas(self, bg="#222", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.status = tk.Label(self, text="Listo", anchor="w"); self.status.pack(fill="x")

        # Eventos canvas
        self.canvas.bind("<ButtonPress-1>", self.on_left_down)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_up)
        self.canvas.bind("<ButtonPress-3>", lambda e: self.canvas.scan_mark(e.x, e.y))
        self.canvas.bind("<B3-Motion>", lambda e: self.canvas.scan_dragto(e.x, e.y, gain=1))
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)  # Windows
        self.canvas.bind("<Button-4>", lambda e: self.set_zoom(self.zoom*1.1)) # Linux up
        self.canvas.bind("<Button-5>", lambda e: self.set_zoom(self.zoom/1.1)) # Linux down

    # ---------- Render ----------
    def render(self):
        if self.display_bgr is None: return
        h, w = self.display_bgr.shape[:2]
        disp = cv2.resize(self.display_bgr, (max(1,int(w*self.zoom)), max(1,int(h*self.zoom))))
        img = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
        self.tk_img = ImageTk.PhotoImage(Image.fromarray(img))
        self.canvas.delete("all")
        self.canvas.create_image(0,0,image=self.tk_img, anchor="nw")
        # Dibuja ROI
        if self.roi_rect is not None:
            x1,y1,x2,y2 = self.roi_rect
            self.canvas.create_rectangle(x1*self.zoom, y1*self.zoom, x2*self.zoom, y2*self.zoom,
                                         outline="#00FF88", width=2)
        self.canvas.config(scrollregion=(0,0, disp.shape[1], disp.shape[0]))

    def show_base(self, bgr):
        self.base_bgr = bgr
        self.display_bgr = bgr.copy()
        self.zoom = min(1.0, 1200/max(bgr.shape[:2]))
        self.render()

    def show_overlay(self, overlay):
        self.display_bgr = overlay
        self.render()

    def set_zoom(self, val):
        if self.base_bgr is None: return
        self.zoom = max(0.25, min(6.0, float(val)))
        self.render()

    # ---------- IO ----------
    def load_images(self):
        paths = filedialog.askopenfilenames(title="Selecciona fotos",
            filetypes=[("Imágenes","*.jpg *.jpeg *.png *.bmp")])
        if not paths:
            self.status["text"] = "No se seleccionaron fotos."; return
        self.img_paths = list(paths)
        bgr = cv2.imread(self.img_paths[0])
        if bgr is None:
            messagebox.showerror("Error","No se pudo abrir la primera foto."); return
        self.show_base(bgr)
        self.scale_lbl["text"] = "Escala: pendiente"
        self.px_per_mm = None
        self.roi_rect = None

    # ---------- Calibración ----------
    def auto_calibrate(self):
        if self.base_bgr is None:
            messagebox.showinfo("Info","Carga primero una foto."); return
        res = auto_detect_coin(self.base_bgr)
        if not res:
            messagebox.showwarning("Auto-calibración","No se detectó la moneda.\nHaz zoom o usa la calibración manual.")
            return
        x,y,r,gold,edge = res
        self.px_per_mm = (2.0*r)/COIN_DIAMETER_MM
        self.scale_lbl["text"] = f"Escala auto: {self.px_per_mm:.3f} px/mm"
        over = self.base_bgr.copy()
        cv2.circle(over,(x,y),r,(0,255,255),3); cv2.circle(over,(x,y),2,(0,0,255),3)
        cv2.putText(over,"Moneda",(x-35,y-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,255,255),2, cv2.LINE_AA)
        self.show_overlay(over)
        self.status["text"] = f"Moneda detectada (gold={gold:.2f}, edge={edge:.2f})."

    def start_calibration(self):
        if self.base_bgr is None:
            messagebox.showinfo("Info","Carga primero una foto."); return
        self.calibrating = True; self.calib_pts.clear()
        self.roi_selecting = False
        self.status["text"] = "Calibración: 2 clics en extremos del DIÁMETRO de la moneda."

    # ---------- ROI ----------
    def start_roi(self):
        if self.base_bgr is None:
            messagebox.showinfo("Info","Carga una foto primero."); return
        self.roi_selecting = True; self.calibrating = False
        self.roi_start = None
        self.status["text"] = "ROI: arrastra con botón izquierdo alrededor de la COPA."

    def clear_roi(self):
        self.roi_selecting = False; self.roi_rect = None; self.render()
        self.status["text"] = "ROI quitada."

    # ---------- Eventos izq ----------
    def on_left_down(self, e):
        if self.base_bgr is None: return
        x = self.canvas.canvasx(e.x)/self.zoom; y = self.canvas.canvasy(e.y)/self.zoom
        if self.roi_selecting:
            self.roi_start = (x,y)
        elif self.calibrating:
            self.calib_pts.append((x,y))
            over = self.display_bgr.copy()
            cv2.circle(over,(int(x),int(y)),4,(0,255,170),2); self.show_overlay(over)

    def on_left_drag(self, e):
        if self.base_bgr is None or not self.roi_selecting or self.roi_start is None: return
        x0,y0 = self.roi_start
        x = self.canvas.canvasx(e.x)/self.zoom; y = self.canvas.canvasy(e.y)/self.zoom
        over = self.base_bgr.copy()
        cv2.rectangle(over,(int(x0),int(y0)),(int(x),int(y)),(0,255,136),2)
        self.show_overlay(over)

    def on_left_up(self, e):
        if self.base_bgr is None: return
        if self.roi_selecting and self.roi_start is not None:
            x0,y0 = self.roi_start
            x = self.canvas.canvasx(e.x)/self.zoom; y = self.canvas.canvasy(e.y)/self.zoom
            self.roi_rect = (min(x0,x), min(y0,y), max(x0,x), max(y0,y))
            self.roi_selecting = False; self.render()
            self.status["text"] = "ROI definida. Detección limitada al rectángulo."
        elif self.calibrating and len(self.calib_pts)==2:
            d_px = distance(self.calib_pts[0], self.calib_pts[1])
            self.px_per_mm = d_px / COIN_DIAMETER_MM
            self.scale_lbl["text"] = f"Escala manual: {self.px_per_mm:.3f} px/mm"
            self.calibrating = False
            self.status["text"] = "Calibración hecha. Usa Vista previa."

    def on_mousewheel(self, e):
        self.set_zoom(self.zoom*(1.1 if e.delta>0 else 1/1.1))

    # ---------- Vista previa y proceso ----------
    def preview_current(self):
        if self.base_bgr is None:
            messagebox.showinfo("Info","Carga una foto primero."); return
        if not self.px_per_mm:
            messagebox.showinfo("Info","Calibra primero (auto o manual)."); return

        min_mm = self.min_mm.get(); max_mm = self.max_mm.get()
        blur = self.blur.get(); p2 = self.hough_p2.get()
        color_pct = float(self.color_pct.get())
        hue_std_max = float(self.hue_std_max.get())
        edge_ring_min = float(self.edge_ring_min.get())
        roi_mask = make_roi_mask(self.base_bgr.shape, self.roi_rect)

        cands, work, (rmin,rmax) = detect_candidates(self.base_bgr, self.px_per_mm, min_mm, max_mm, blur, p2, roi_mask)
        found = quality_filters(work, cands, color_pct, hue_std_max, edge_ring_min, roi_mask)

        used_fallback = False
        if not found and self.modo_robusto.get():
            found = detect_by_contours(self.base_bgr, self.px_per_mm, min_mm, max_mm, blur, roi_mask)
            used_fallback = bool(found)

        over = self.base_bgr.copy()
        for (x,y,r) in found:
            dmm = (2*r)/self.px_per_mm
            cv2.circle(over,(x,y),r,(0,255,0),2); cv2.circle(over,(x,y),2,(0,0,255),3)
            cv2.putText(over,f"{round(dmm)}mm",(x-20,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
        self.show_overlay(over)

        metodo = "Contornos" if used_fallback else "Hough+filtros"
        self.status["text"] = f"Vista previa: {len(found)} frutos. R(px) {rmin}-{rmax}. Método: {metodo}."

    def process_all(self):
        if not self.img_paths:
            messagebox.showinfo("Info","Carga fotos primero."); return
        if not self.px_per_mm:
            messagebox.showinfo("Info","Calibra primero."); return

        min_mm = self.min_mm.get(); max_mm = self.max_mm.get()
        blur = self.blur.get(); p2 = self.hough_p2.get()
        color_pct = float(self.color_pct.get())
        hue_std_max = float(self.hue_std_max.get())
        edge_ring_min = float(self.edge_ring_min.get())
        muestreados = max(1,int(self.arboles_muestreados.get()))
        total_arboles = max(1,int(self.num_arboles.get()))
        rho = float(self.densidad.get()); k = float(self.factor_forma.get())

        out = ensure_dir("salida_aforo")
        rows, diams = [], []
        total_count = 0

        # Nota: ROI se aplica solo si la resolución coincide con la base (misma foto).
        roi_mask_base = make_roi_mask(self.base_bgr.shape, self.roi_rect) if self.base_bgr is not None else None

        for p in self.img_paths:
            bgr = cv2.imread(p)
            if bgr is None: continue
            roi_mask = roi_mask_base if (roi_mask_base is not None and bgr.shape==self.base_bgr.shape) else None

            cands, work, _ = detect_candidates(bgr, self.px_per_mm, min_mm, max_mm, blur, p2, roi_mask)
            found = quality_filters(work, cands, color_pct, hue_std_max, edge_ring_min, roi_mask)
            used_fallback = False
            if not found and self.modo_robusto.get():
                found = detect_by_contours(bgr, self.px_per_mm, min_mm, max_mm, blur, roi_mask)
                used_fallback = bool(found)

            over = bgr.copy()
            name = os.path.splitext(os.path.basename(p))[0]
            for i,(x,y,r) in enumerate(found, start=1):
                dmm = (2*r)/self.px_per_mm
                diams.append(dmm)
                rows.append({"foto": os.path.basename(p), "fruto_id": f"{name}_{i}",
                             "x_px": x, "y_px": y, "radio_px": r, "diametro_mm": round(dmm,2),
                             "metodo": "contornos" if used_fallback else "hough_filtros"})
                cv2.circle(over,(x,y),r,(0,255,0),2); cv2.circle(over,(x,y),2,(0,0,255),3)
                cv2.putText(over,f"{round(dmm)}mm",(x-20,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2,cv2.LINE_AA)
            cv2.imwrite(os.path.join(out, f"{name}_detectado.jpg"), over)
            total_count += len(found)

        # Resumen de parcela
        resumen = None
        if diams:
            avg_d = float(np.mean(diams))
            vol_cm3 = (math.pi/6.0)*((avg_d/10.0)**3)
            peso_g = rho*vol_cm3*k
            frutos_arbol = total_count/muestreados
            total_frutos = frutos_arbol*total_arboles
            total_kg = (total_frutos*peso_g)/1000.0
            resumen = {"fotos": len(self.img_paths), "frutos_detectados": total_count,
                       "arboles_muestreados": muestreados, "frutos_por_arbol": round(frutos_arbol,2),
                       "num_arboles_parcela": total_arboles, "diametro_medio_mm": round(avg_d,2),
                       "peso_medio_g": round(peso_g,1), "total_frutos_parcela": int(round(total_frutos)),
                       "total_kg_estimado": round(total_kg,1), "densidad": rho, "factor_forma": k}

        # CSVs
        csv_path = resumen_path = None
        if rows:
            csv_path = os.path.join(out,"resultados_aforo.csv"); pd.DataFrame(rows).to_csv(csv_path, index=False, sep=";")
        if resumen:
            resumen_path = os.path.join(out,"resumen_parcela.csv"); pd.DataFrame([resumen]).to_csv(resumen_path, index=False, sep=";")

        msg = (f"Procesadas {len(self.img_paths)} fotos.\n"
               f"Frutos detectados: {total_count}\n"
               f"{'CSV frutos: '+csv_path if csv_path else ''}\n"
               f"{'CSV resumen: '+resumen_path if resumen_path else ''}")
        self.status["text"] = msg
        messagebox.showinfo("Resultado", msg)

# ---------------- Main ----------------------
if __name__ == "__main__":
    App().mainloop()
