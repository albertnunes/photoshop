import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk   # pip install pillow
import imageio.v3 as iio         # pip install imageio
import os

# ─────────────────────────────────────────────────────────────────────────────
# PALETA DE CORES — tema escuro estilo editor profissional
# ─────────────────────────────────────────────────────────────────────────────
BG        = "#fcb421"   # fundo geral
PANEL     = "#1094ab"   # painéis laterais
CARD      = "#64c4d2"   # cards de seção
ACCENT    = "#1094ab"   # azul-claro para destaques e botões
ACCENT2   = "#1094ab"   # azul-claro para títulos e abas selecionadas
TEXT      = "#000000"   # texto principal
TEXT_DIM  = "#000000"   # texto secundário
BTN_BG    = "#fcb421"   # botão primário
BTN_HV    = "#ffffff"   # hover do botão primário
BTN2_BG   = "#fcb421"   # botão secundário
BTN2_HV   = "#ffffff"   # hover do botão secundário
SEP       = "#fcb421"   # separadores


# ─────────────────────────────────────────────────────────────────────────────
# FUNÇÕES DE PROCESSAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def translacao(img, dx, dy):
    """Translação com wrap-around circular usando np.roll."""
    res = np.roll(img, dy, axis=0)
    res = np.roll(res,  dx, axis=1)
    return res


def rotacao(img, angulo):
    """Rotação em torno do centro com mapeamento inverso + interpolação bilinear."""
    H, W = img.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    rad = np.deg2rad(-angulo)
    cos_a, sin_a = np.cos(rad), np.sin(rad)

    ys_out, xs_out = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs_src = cos_a * (xs_out - cx) - sin_a * (ys_out - cy) + cx
    ys_src = sin_a * (xs_out - cx) + cos_a * (ys_out - cy) + cy

    x0 = np.floor(xs_src).astype(int);  y0 = np.floor(ys_src).astype(int)
    wa = xs_src - x0;                   wb = ys_src - y0
    x0 = np.clip(x0, 0, W-1);  x1 = np.clip(x0+1, 0, W-1)
    y0 = np.clip(y0, 0, H-1);  y1 = np.clip(y0+1, 0, H-1)

    res = np.zeros_like(img, dtype=np.float64)
    for c in range(img.shape[2]):
        res[:,:,c] = (img[y0,x0,c]*(1-wa)*(1-wb) + img[y0,x1,c]*wa*(1-wb) +
                      img[y1,x0,c]*(1-wa)*wb      + img[y1,x1,c]*wa*wb)
    return np.clip(res, 0, 255).astype(np.uint8)


def escala(img, fator):
    """Escala com bilinear + crop/pad centralizado para manter dimensões."""
    if fator <= 0: return img
    H, W = img.shape[:2]
    nH, nW = max(1, int(round(H*fator))), max(1, int(round(W*fator)))

    ys, xs = np.meshgrid(np.linspace(0,H-1,nH), np.linspace(0,W-1,nW), indexing='ij')
    x0 = np.floor(xs).astype(int);  y0 = np.floor(ys).astype(int)
    wa = xs-x0;  wb = ys-y0
    x0 = np.clip(x0,0,W-1); x1 = np.clip(x0+1,0,W-1)
    y0 = np.clip(y0,0,H-1); y1 = np.clip(y0+1,0,H-1)

    redim = np.zeros((nH,nW,img.shape[2]), dtype=np.float64)
    for c in range(img.shape[2]):
        redim[:,:,c] = (img[y0,x0,c]*(1-wa)*(1-wb) + img[y0,x1,c]*wa*(1-wb) +
                        img[y1,x0,c]*(1-wa)*wb      + img[y1,x1,c]*wa*wb)
    redim = np.clip(redim, 0, 255).astype(np.uint8)

    if fator >= 1:
        sy, sx = (nH-H)//2, (nW-W)//2
        return redim[sy:sy+H, sx:sx+W]
    else:
        result = np.zeros_like(img)
        sy, sx = (H-nH)//2, (W-nW)//2
        result[sy:sy+nH, sx:sx+nW] = redim
        return result


def inversa(img):
    """Negativo: f(r) = 255 - r."""
    return (255 - img.astype(np.int16)).clip(0,255).astype(np.uint8)


def transformacao_log(img):
    """Log: f(r) = c * log(1 + r)."""
    c = 255.0 / np.log1p(255.0)
    return np.clip(c * np.log1p(img.astype(np.float64)), 0, 255).astype(np.uint8)


def gamma(img, g):
    """Gamma: f(r) = (r/255)^g * 255."""
    if g <= 0: return img
    return np.clip(np.power(img/255.0, g)*255.0, 0, 255).astype(np.uint8)


def modulacao_contraste(img, a, b):
    """Stretching linear do intervalo [a,b] para [0,255]."""
    if a >= b: return img
    return np.clip((img.astype(np.float64)-a)/(b-a)*255.0, 0, 255).astype(np.uint8)


def solarizacao(img, limiar):
    """Solarização (Efeito Sabattier): inverte pixels acima do limiar."""
    arr = img.astype(np.int16)
    return np.where(arr >= limiar, 255 - arr, arr).clip(0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# WIDGETS AUXILIARES
# ─────────────────────────────────────────────────────────────────────────────

def estilizar_botao(btn, cor=BTN_BG, hover=BTN_HV):
    btn.configure(bg=cor, fg=TEXT, relief="flat", cursor="hand2",
                  activebackground=hover, activeforeground=TEXT,
                  bd=0, padx=12, pady=6)
    btn.bind("<Enter>", lambda e: btn.configure(bg=hover))
    btn.bind("<Leave>", lambda e: btn.configure(bg=cor))


def separador(parent):
    return tk.Frame(parent, height=1, bg=SEP)


# ─────────────────────────────────────────────────────────────────────────────
# APLICAÇÃO PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

class EditorApp:
    """
    Interface gráfica do editor de imagens.

    Layout:
    ┌──────────────────────────────────────────────────┐
    │  TOPBAR  (título + Abrir / Salvar)                │
    ├──────────────┬───────────────────────────────────┤
    │  PAINEL      │  ÁREA DE VISUALIZAÇÃO              │
    │  ESQUERDO    │  original  │  resultado            │
    │  (controles) │            │                       │
    └──────────────┴───────────────────────────────────┘
    """

    def __init__(self, root):
        self.root = root
        self.root.title("Editor de Imagens - Albert Lincoln Lage Nunes - 18008738 - SCC5830")
        self.root.configure(bg=BG)
        self.root.minsize(1100, 660)

        self.img_original = None   # ndarray carregado do disco
        self.img_resultado = None  # ndarray após aplicação de filtros

        self._build_ui()

    # ── Construção da UI ──────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_topbar()
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        main.columnconfigure(1, weight=1)
        main.rowconfigure(0, weight=1)
        self._build_painel_esquerdo(main)
        self._build_area_visualizacao(main)

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=CARD, pady=10)
        bar.pack(fill="x")
        tk.Label(bar, text="◈  ADOUBÊ PHOTOSHOP",
                 font=("Courier", 16, "bold"), fg=ACCENT, bg=CARD).pack(side="left", padx=20)
        tk.Label(bar, text="CREATOR - ALBERT NUNES · NUSP 18008738 · SCC5830",
                 font=("Courier", 10), fg=TEXT_DIM, bg=CARD).pack(side="left")

        btn_salvar = tk.Button(bar, text="💾  Salvar resultado", command=self.salvar_resultado)
        estilizar_botao(btn_salvar, BTN2_BG, BTN2_HV)
        btn_salvar.pack(side="right", padx=10)

        btn_abrir = tk.Button(bar, text="📂  Abrir imagem", command=self.abrir_imagem)
        estilizar_botao(btn_abrir)
        btn_abrir.pack(side="right", padx=(10, 0))

    def _build_painel_esquerdo(self, parent):
        frame = tk.Frame(parent, bg=PANEL, width=300)
        frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        frame.pack_propagate(False)

        tk.Label(frame, text="TRANSFORMAÇÕES", font=("Courier", 9),
                 fg=TEXT_DIM, bg=PANEL).pack(anchor="w", padx=16, pady=(14, 4))
        separador(frame).pack(fill="x", padx=8, pady=(0, 8))

        # Botão reset fixado no fundo ANTES do notebook para que o pack reserve espaço
        separador(frame).pack(side="bottom", fill="x", padx=8, pady=(8, 0))
        btn_reset = tk.Button(frame, text="↺  Resetar imagem", command=self.resetar)
        estilizar_botao(btn_reset, CARD, SEP)
        btn_reset.pack(side="bottom", fill="x", padx=16, pady=(0, 12))

        # Configurar estilo das abas
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=PANEL, borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=CARD, foreground=TEXT_DIM,
                        font=("Courier", 9, "bold"), padding=[10, 5])
        style.map("Dark.TNotebook.Tab",
                  background=[("selected", ACCENT2)],
                  foreground=[("selected", TEXT)])

        nb = ttk.Notebook(frame, style="Dark.TNotebook")
        nb.pack(fill="both", expand=True, padx=8, pady=4)

        tab_geo = tk.Frame(nb, bg=PANEL)
        nb.add(tab_geo, text=" GEO ")
        self._build_tab_geometricas(tab_geo)

        tab_int = tk.Frame(nb, bg=PANEL)
        nb.add(tab_int, text=" INT ")
        self._build_tab_intensidade(tab_int)

    def _card(self, parent, titulo):
        """Container estilizado com título para cada filtro."""
        outer = tk.Frame(parent, bg=CARD, pady=8, padx=10)
        outer.pack(fill="x", pady=(0, 6))
        tk.Label(outer, text=titulo, font=("Courier", 9, "bold"),
                 fg=ACCENT, bg=CARD).pack(anchor="w", pady=(0, 6))
        return outer

    def _build_tab_geometricas(self, tab):
        f = tk.Frame(tab, bg=PANEL)
        f.pack(fill="both", expand=True, padx=4, pady=4)

        c = self._card(f, "▸ Translação")
        self.var_dx = self._slider_spin(c, "dx (px)", -500, 500, 0)
        self.var_dy = self._slider_spin(c, "dy (px)", -500, 500, 0)
        self._btn_aplicar(c, self._aplicar_translacao)

        c = self._card(f, "▸ Rotação")
        self.var_rot = self._slider_spin(c, "ângulo (°)", -180, 180, 0)
        self._btn_aplicar(c, self._aplicar_rotacao)

        c = self._card(f, "▸ Escala / Crop")
        self.var_esc = self._slider_spin(c, "fator", 0.1, 3.0, 1.0, res=0.05, float_=True)
        self._btn_aplicar(c, self._aplicar_escala)

    def _build_tab_intensidade(self, tab):
        # Canvas + scrollbar para a aba ter scroll vertical
        canvas = tk.Canvas(tab, bg=PANEL, highlightthickness=0)
        sb = tk.Scrollbar(tab, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=sb.set)
        sb.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        f = tk.Frame(canvas, bg=PANEL)
        win_id = canvas.create_window((0, 0), window=f, anchor="nw")
        f.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfig(win_id, width=e.width))
        # Scroll com roda do mouse
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(-1 if e.delta > 0 else 1, "units"))

        c = self._card(f, "▸ Inversa (negativo)")
        self._btn_aplicar(c, self._aplicar_inversa, "Aplicar Inversa")

        c = self._card(f, "▸ Logarítmica")
        self._btn_aplicar(c, self._aplicar_log, "Aplicar Log")

        c = self._card(f, "▸ Gamma")
        self.var_gamma = self._slider_spin(c, "γ", 0.1, 5.0, 1.0, res=0.05, float_=True)
        self._btn_aplicar(c, self._aplicar_gamma)

        c = self._card(f, "▸ Modulação de Contraste")
        self.var_cont_a = self._slider_spin(c, "a (inf)", 0, 254, 0)
        self.var_cont_b = self._slider_spin(c, "b (sup)", 1, 255, 255)
        self._btn_aplicar(c, self._aplicar_contraste)

        c = self._card(f, "▸ Solarização ✦")
        self.var_sol = self._slider_spin(c, "limiar", 0, 255, 128)
        self._btn_aplicar(c, self._aplicar_solarizacao)

    def _build_area_visualizacao(self, parent):
        frame = tk.Frame(parent, bg=BG)
        frame.grid(row=0, column=1, sticky="nsew")
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(1, weight=1)

        for col, (txt, cor) in enumerate([("ORIGINAL", ACCENT2), ("RESULTADO", ACCENT)]):
            tk.Label(frame, text=txt, font=("Courier", 10, "bold"),
                     fg=cor, bg=BG).grid(row=0, column=col, pady=(6, 2))

        self.canvas_orig = self._make_canvas(frame)
        self.canvas_orig.grid(row=1, column=0, sticky="nsew", padx=(0, 4))

        self.canvas_res = self._make_canvas(frame)
        self.canvas_res.grid(row=1, column=1, sticky="nsew", padx=(4, 0))

        self.status_var = tk.StringVar(value="Abra uma imagem para começar.")
        tk.Label(frame, textvariable=self.status_var,
                 font=("Courier", 9), fg=TEXT_DIM, bg=BG, anchor="w"
                 ).grid(row=2, column=0, columnspan=2, sticky="ew", pady=(4, 0))

    def _make_canvas(self, parent):
        c = tk.Canvas(parent, bg=CARD, highlightthickness=1,
                      highlightbackground=SEP, cursor="crosshair")
        c.bind("<Configure>", lambda e: self._redraw_previews())
        return c

    # ── Widgets auxiliares ────────────────────────────────────────────────────

    def _slider_spin(self, parent, label_txt, mn, mx, ini, res=1, float_=False):
        """Cria par Slider+Spinbox sincronizados. Retorna a variável compartilhada."""
        var = tk.DoubleVar(value=ini) if float_ else tk.IntVar(value=ini)
        tk.Label(parent, text=label_txt, font=("Courier", 8),
                 fg=TEXT_DIM, bg=CARD).pack(anchor="w")
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(0, 4))
        tk.Scale(row, from_=mn, to=mx, orient="horizontal", variable=var,
                 resolution=res, bg=CARD, fg=TEXT, troughcolor=BG,
                 highlightthickness=0, showvalue=False, activebackground=ACCENT
                 ).pack(side="left", fill="x", expand=True)
        tk.Spinbox(row, from_=mn, to=mx, increment=res, textvariable=var,
                   width=6, format="%.2f" if float_ else None,
                   bg=BG, fg=TEXT, insertbackground=TEXT,
                   buttonbackground=CARD, relief="flat", font=("Courier", 9)
                   ).pack(side="right", padx=(4, 0))
        return var

    def _btn_aplicar(self, parent, cmd, text="▶  Aplicar"):
        btn = tk.Button(parent, text=text, command=cmd)
        estilizar_botao(btn)
        btn.pack(fill="x", pady=(4, 0))

    # ── Ações de arquivo ──────────────────────────────────────────────────────

    def abrir_imagem(self):
        """Abre diálogo para carregar imagem do disco."""
        tipos = [("Imagens", "*.png *.jpg *.jpeg *.bmp *.tiff *.webp"), ("Todos", "*.*")]
        caminho = filedialog.askopenfilename(filetypes=tipos)
        if not caminho: return
        try:
            img = iio.imread(caminho)
            if img.ndim == 3 and img.shape[2] == 4:
                img = img[:, :, :3]
            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)
            self.img_original = img.astype(np.uint8)
            self.img_resultado = self.img_original.copy()
            H, W = img.shape[:2]
            self.status_var.set(f"✔ {os.path.basename(caminho)}  —  {W}×{H} px")
            self._redraw_previews()
        except Exception as e:
            messagebox.showerror("Erro", f"Não foi possível carregar:\n{e}")

    def salvar_resultado(self):
        """Salva img_resultado em arquivo escolhido pelo usuário."""
        if self.img_resultado is None:
            messagebox.showwarning("Atenção", "Nenhuma imagem processada.")
            return
        tipos = [("PNG", "*.png"), ("JPEG", "*.jpg"), ("Todos", "*.*")]
        caminho = filedialog.asksaveasfilename(defaultextension=".png", filetypes=tipos)
        if caminho:
            iio.imwrite(caminho, self.img_resultado)
            self.status_var.set(f"✔ Salvo em: {caminho}")

    def resetar(self):
        """Restaura img_resultado para o original e reseta todos os sliders."""
        if self.img_original is None: return
        self.img_resultado = self.img_original.copy()
        self.var_dx.set(0)
        self.var_dy.set(0)
        self.var_rot.set(0)
        self.var_esc.set(1.0)
        self.var_gamma.set(1.0)
        self.var_cont_a.set(0)
        self.var_cont_b.set(255)
        self.var_sol.set(128)
        self.status_var.set("↺ Imagem resetada para o original.")
        self._redraw_previews()

    # ── Aplicação de filtros ──────────────────────────────────────────────────

    def _img_base(self):
        """Retorna img_resultado atual (ou avisa se nenhuma foi carregada)."""
        if self.img_resultado is None:
            messagebox.showwarning("Atenção", "Carregue uma imagem primeiro.")
            return None
        return self.img_resultado

    def _finalizar(self, resultado, msg):
        """Atualiza resultado, preview e barra de status."""
        self.img_resultado = resultado
        self.status_var.set(f"✔ {msg}")
        self._redraw_previews()

    def _aplicar_translacao(self):
        img = self._img_base()
        if img is None: return
        dx, dy = int(self.var_dx.get()), int(self.var_dy.get())
        self._finalizar(translacao(img, dx, dy), f"Translação: dx={dx}, dy={dy}")

    def _aplicar_rotacao(self):
        img = self._img_base()
        if img is None: return
        ang = float(self.var_rot.get())
        self._finalizar(rotacao(img, ang), f"Rotação: {ang}°")

    def _aplicar_escala(self):
        img = self._img_base()
        if img is None: return
        f = float(self.var_esc.get())
        self._finalizar(escala(img, f), f"Escala: fator={f:.2f}")

    def _aplicar_inversa(self):
        img = self._img_base()
        if img is None: return
        self._finalizar(inversa(img), "Inversa (negativo)")

    def _aplicar_log(self):
        img = self._img_base()
        if img is None: return
        self._finalizar(transformacao_log(img), "Transformação logarítmica")

    def _aplicar_gamma(self):
        img = self._img_base()
        if img is None: return
        g = float(self.var_gamma.get())
        self._finalizar(gamma(img, g), f"Gamma: γ={g:.2f}")

    def _aplicar_contraste(self):
        img = self._img_base()
        if img is None: return
        a, b = int(self.var_cont_a.get()), int(self.var_cont_b.get())
        if a >= b:
            messagebox.showwarning("Atenção", "'a' deve ser menor que 'b'.")
            return
        self._finalizar(modulacao_contraste(img, a, b), f"Contraste: [{a},{b}] → [0,255]")

    def _aplicar_solarizacao(self):
        img = self._img_base()
        if img is None: return
        lim = int(self.var_sol.get())
        self._finalizar(solarizacao(img, lim), f"Solarização: limiar={lim}")

    # ── Renderização dos previews ─────────────────────────────────────────────

    def _fit_image(self, img, canvas):
        """Redimensiona img para caber no canvas mantendo proporção de aspecto."""
        cw = canvas.winfo_width()  or 400
        ch = canvas.winfo_height() or 400
        H, W = img.shape[:2]
        ratio = min(cw / W, ch / H, 1.0)
        nW, nH = max(1, int(W * ratio)), max(1, int(H * ratio))
        pil_img = Image.fromarray(img).resize((nW, nH), Image.LANCZOS)
        return ImageTk.PhotoImage(pil_img)

    def _draw_on_canvas(self, canvas, photo):
        """Centraliza e exibe a imagem no canvas."""
        canvas.delete("all")
        cw = canvas.winfo_width()  or 400
        ch = canvas.winfo_height() or 400
        canvas.create_image(cw // 2, ch // 2, anchor="center", image=photo)
        canvas.image = photo   # mantém referência — evita garbage collection do Tk

    def _redraw_previews(self):
        """Atualiza os dois painéis de visualização."""
        if self.img_original is not None:
            ph = self._fit_image(self.img_original, self.canvas_orig)
            self._draw_on_canvas(self.canvas_orig, ph)
        if self.img_resultado is not None:
            ph = self._fit_image(self.img_resultado, self.canvas_res)
            self._draw_on_canvas(self.canvas_res, ph)


# ─────────────────────────────────────────────────────────────────────────────
# PONTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def main():
    root = tk.Tk()
    root.geometry("1200x700")
    EditorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()