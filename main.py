"""
Filtro com as mãos
------------------
Detecta as mãos pela webcam, identifica cada dedo (polegar, indicador, médio,
anelar e mínimo) e liga as pontas dos dedos LEVANTADOS, formando uma figura.
A imagem dentro da figura recebe um filtro que depende da forma:

    2 dedos   Círculo             -> Lupa (zoom)
    3 dedos   Triângulo           -> Pontilhado P&B
    4 dedos   Quadrado            -> Pontilhado colorido
    4 dedos   Retângulo deitado   -> Cartoon
    4 dedos   Retângulo em pé     -> Azul
    5 dedos   Pentágono           -> Esboço a lápis
    6 dedos   Hexágono            -> Térmico
    7 dedos   Heptágono           -> Pixelado
    8 dedos   Octógono            -> Negativo
    9 dedos   Eneágono            -> Sépia
    10 dedos  Decágono            -> Neon

Interface em Full HD (1920x1080): a câmera é lida numa thread própria e a
detecção das mãos roda de forma assíncrona, então o vídeo não trava esperando
o modelo. Os pontos são suavizados com o filtro One Euro.

Teclas:  F tela cheia   H mostra/esconde o painel   Q ou ESC sai
"""

import math
import os
import threading
import time
import urllib.request
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from PIL import Image, ImageDraw, ImageFilter, ImageFont

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# ----------------------------------------------------------------------------
# Configuração
# ----------------------------------------------------------------------------

LARGURA, ALTURA = 1920, 1080   # resolução da interface
DETECCAO_L = 640               # largura da imagem enviada ao detector (mais rápido)
PAINEL_L = 420                 # largura do painel lateral
MARGEM = 28                    # distância do painel até a borda
SS = 3                         # super-amostragem dos desenhos do painel (linhas mais suaves)

# One Euro: menos tremida parado (MIN_CUTOFF menor) x menos atraso em movimento (BETA maior)
MIN_CUTOFF = 1.4
BETA = 0.008
# Quantos quadros seguidos uma forma precisa aparecer para trocar de filtro
SHAPE_STABLE_FRAMES = 4
# Duração do fade quando o filtro muda (segundos)
FADE = 0.25
# Razão lado maior / lado menor abaixo da qual 4 pontos contam como "quadrado"
SQUARE_RATIO = 1.35
# Dedo levantado: ponta X vezes mais longe do pulso que a articulação do meio.
# Dois limites (subir / descer) evitam que o dedo fique piscando.
DEDO_SOBE, DEDO_DESCE = 1.20, 1.08
# Polegar: distância da ponta até a base do indicador, em tamanhos de palma
POLEGAR_SOBE, POLEGAR_DESCE = 0.55, 0.42


def hex_bgr(h):
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


# (nome, ponta, articulação do meio) de cada dedo
DEDOS = [
    ("Polegar", 4, 3),
    ("Indicador", 8, 6),
    ("Médio", 12, 10),
    ("Anelar", 16, 14),
    ("Mínimo", 20, 18),
]
DEDOS_CURTO = ["Pol", "Ind", "Méd", "Anel", "Mín"]
COR_DEDO = [hex_bgr(c) for c in ("#fb7185", "#fdba74", "#fde047", "#86efac", "#93c5fd")]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# ----------------------------------------------------------------------------
# Filtros  (recebem e devolvem uma imagem BGR do mesmo tamanho)
# ----------------------------------------------------------------------------

GRADE = 48  # recorte alinhado a esta grade para os padrões não "nadarem" ao mover a mão


def alinhado(fn):
    """Marca filtros de padrão fixo (bolinhas, pixels) para usarem recorte alinhado."""
    fn.alinhado = True
    return fn


@lru_cache(maxsize=8)
def _distancias(cell, h, w):
    """Distância de cada pixel ao centro da sua célula (cacheada para o tamanho máximo)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dy = (yy % cell) - (cell - 1) / 2.0
    dx = (xx % cell) - (cell - 1) / 2.0
    return np.sqrt(dx * dx + dy * dy)


def _bolinhas(img, cell, raio_por_celula, cor_bolinha, cor_fundo):
    """Desenha uma bolinha anti-aliased por célula e mistura cor_bolinha sobre cor_fundo."""
    h, w = img.shape[:2]
    sh, sw = raio_por_celula.shape
    raio = cv2.resize(raio_por_celula, (sw * cell, sh * cell), interpolation=cv2.INTER_NEAREST)[:h, :w]
    dist = _distancias(cell, ALTURA + 2 * GRADE, LARGURA + 2 * GRADE)[:h, :w]
    cobertura = np.clip(raio - dist + 0.5, 0.0, 1.0)
    return cv2.blendLinear(cor_bolinha, cor_fundo, cobertura, 1.0 - cobertura)


def _celulas(img, cell):
    h, w = img.shape[:2]
    return cv2.resize(img, (-(-w // cell), -(-h // cell)), interpolation=cv2.INTER_AREA)


@alinhado
def filtro_pontilhado_colorido(img, cell=12):
    """Bolinhas coloridas sobre fundo escuro; tamanho pelo brilho."""
    h, w = img.shape[:2]
    small = _celulas(img, cell)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    raio = (0.2 + 0.62 * np.sqrt(lum)) * cell / 2.0 * 1.2
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * 1.5, 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * 1.15, 0, 255)
    small = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    cores = cv2.resize(small, (small.shape[1] * cell, small.shape[0] * cell),
                       interpolation=cv2.INTER_NEAREST)[:h, :w]
    return _bolinhas(img, cell, raio, cores, np.full_like(img, 18))


@alinhado
def filtro_pontilhado_pb(img, cell=8):
    """Retícula de jornal: bolinhas pretas no papel branco; tamanho pela sombra."""
    small = _celulas(img, cell)
    lum = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    lum = np.clip((lum - 0.5) * 1.3 + 0.5, 0, 1)
    raio = np.sqrt(1.0 - lum) * cell / 2.0 * 1.3
    return _bolinhas(img, cell, raio, np.full_like(img, 20), np.full_like(img, 242))


def filtro_cartoon(img):
    """Cores chapadas + contornos finos."""
    h, w = img.shape[:2]
    small = cv2.pyrDown(cv2.pyrDown(img)) if min(h, w) > 64 else img
    for _ in range(2):
        small = cv2.bilateralFilter(small, 7, 60, 7)
    cor = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    cor = (cor // 40) * 40 + 20
    gray = cv2.medianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 5)
    bordas = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY, 9, 5)
    return cv2.bitwise_and(cor, cor, mask=bordas)


def filtro_azul(img):
    """Tom monocromático azul (estilo cianotipia)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lut = np.zeros((256, 1, 3), np.uint8)
    g = np.arange(256, dtype=np.float32)
    lut[:, 0, 0] = np.clip(g * 1.1 + 60, 0, 255)
    lut[:, 0, 1] = np.clip(g * 0.75 + 10, 0, 255)
    lut[:, 0, 2] = np.clip(g * 0.35, 0, 255)
    return cv2.LUT(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), lut)


def filtro_lupa(img, zoom=2.0):
    """Amplia o centro da região."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(h / zoom)), max(1, int(w / zoom))
    y0, x0 = (h - ch) // 2, (w - cw) // 2
    return cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (w, h), interpolation=cv2.INTER_CUBIC)


def filtro_lapis(img):
    """Desenho a lápis em preto e branco."""
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    inv = cv2.resize(255 - gray, (max(1, w // 2), max(1, h // 2)), interpolation=cv2.INTER_AREA)
    blur = cv2.resize(cv2.GaussianBlur(inv, (0, 0), 5), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(cv2.divide(gray, 255 - blur, scale=256), cv2.COLOR_GRAY2BGR)


def filtro_termico(img):
    """Câmera térmica falsa (mapa de cores pelo brilho)."""
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)


@alinhado
def filtro_pixelado(img, bloco=24):
    h, w = img.shape[:2]
    small = cv2.resize(img, (-(-w // bloco), -(-h // bloco)), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (small.shape[1] * bloco, small.shape[0] * bloco),
                      interpolation=cv2.INTER_NEAREST)[:h, :w]


def filtro_negativo(img):
    return cv2.bitwise_not(img)


_SEPIA = np.array([[0.131, 0.534, 0.272],
                   [0.168, 0.686, 0.349],
                   [0.189, 0.769, 0.393]], dtype=np.float32)  # BGR


def filtro_sepia(img):
    return cv2.transform(img, _SEPIA)


def filtro_neon(img):
    """Contornos finos e brilhantes em cores do arco-íris sobre fundo preto."""
    h, w = img.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    bordas = cv2.Canny(gray, 40, 110)
    rampa = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    linhas = cv2.bitwise_and(cv2.applyColorMap(rampa, cv2.COLORMAP_HSV),
                             cv2.applyColorMap(rampa, cv2.COLORMAP_HSV), mask=bordas)
    pequeno = cv2.resize(linhas, (max(1, w // 4), max(1, h // 4)), interpolation=cv2.INTER_AREA)
    brilho = cv2.resize(cv2.GaussianBlur(pequeno, (0, 0), 2.5), (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.add(linhas, cv2.convertScaleAbs(brilho, alpha=3.0))


# chave, nome da forma, como fazer, nome do filtro, função, cor
FORMAS = [
    ("circulo",    "Círculo",           "2 dedos · polegar + indicador",  "Lupa",           filtro_lupa,                hex_bgr("#7dd3fc")),
    ("triangulo",  "Triângulo",         "3 dedos",                        "Pontilhado P&B", filtro_pontilhado_pb,       hex_bgr("#f5f5f4")),
    ("quadrado",   "Quadrado",          "4 dedos · moldura quadrada",     "Pontilhado cor", filtro_pontilhado_colorido, hex_bgr("#fbbf24")),
    ("horizontal", "Retângulo deitado", "4 dedos · moldura larga",        "Cartoon",        filtro_cartoon,             hex_bgr("#86efac")),
    ("vertical",   "Retângulo em pé",   "4 dedos · moldura alta",         "Azul",           filtro_azul,                hex_bgr("#60a5fa")),
    ("pentagono",  "Pentágono",         "5 dedos · uma mão aberta",       "Lápis",          filtro_lapis,               hex_bgr("#d6d3d1")),
    ("hexagono",   "Hexágono",          "6 dedos · 3 em cada mão",        "Térmico",        filtro_termico,             hex_bgr("#fb923c")),
    ("heptagono",  "Heptágono",         "7 dedos · mão aberta + 2",       "Pixelado",       filtro_pixelado,            hex_bgr("#e879f9")),
    ("octogono",   "Octógono",          "8 dedos · 4 em cada mão",        "Negativo",       filtro_negativo,            hex_bgr("#f9a8d4")),
    ("eneagono",   "Eneágono",          "9 dedos",                        "Sépia",          filtro_sepia,               hex_bgr("#d4a373")),
    ("decagono",   "Decágono",          "10 dedos · duas mãos abertas",   "Neon",           filtro_neon,                hex_bgr("#a78bfa")),
]
FORMA = {f[0]: f for f in FORMAS}
INDICE_FORMA = {f[0]: i for i, f in enumerate(FORMAS)}
LADOS = {"triangulo": 3, "pentagono": 5, "hexagono": 6, "heptagono": 7,
         "octogono": 8, "eneagono": 9, "decagono": 10}
FORMA_POR_LADOS = {n: k for k, n in LADOS.items()}


# ----------------------------------------------------------------------------
# Rastreamento: suavização One Euro e estado dos dedos
# ----------------------------------------------------------------------------

class OneEuro:
    """Filtro One Euro vetorizado: suave quando parado, rápido quando em movimento."""

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(dt, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return self.x
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = self._alpha(dt, self.d_cutoff)
        self.dx = a_d * dx + (1 - a_d) * self.dx
        velocidade = np.linalg.norm(self.dx, axis=-1, keepdims=True)
        a = self._alpha(dt, self.min_cutoff + self.beta * velocidade)
        self.x = a * x + (1 - a) * self.x
        self.t = t
        return self.x


def _dist(a, b):
    return float(np.linalg.norm(a - b))


class Mao:
    def __init__(self):
        self.suave = OneEuro()
        self.pts = None
        self.lado = ""
        self.estados = [False] * 5

    def atualizar(self, pts, lado, t):
        self.pts = self.suave(pts, t)
        self.lado = lado
        self.estados = dedos_levantados(self.pts, self.estados)


def dedos_levantados(pts, antes):
    """Retorna [polegar, indicador, médio, anelar, mínimo] como True/False (com histerese)."""
    palma = max(_dist(pts[0], pts[9]), 1e-6)

    s = _dist(pts[4], pts[5]) / palma
    r = _dist(pts[4], pts[17]) / max(_dist(pts[3], pts[17]), 1e-6)
    if antes[0]:
        polegar = s > POLEGAR_DESCE and r > 1.0
    else:
        polegar = s > POLEGAR_SOBE and r > 1.05
    estados = [polegar]

    for i, (_, ponta, meio) in enumerate(DEDOS[1:], start=1):
        razao = _dist(pts[ponta], pts[0]) / max(_dist(pts[meio], pts[0]), 1e-6)
        estados.append(razao > (DEDO_DESCE if antes[i] else DEDO_SOBE))
    return estados


# ----------------------------------------------------------------------------
# Geometria das formas
# ----------------------------------------------------------------------------

def ordenar_poligono(pts):
    """Ordena pontos pelo ângulo em volta do centro (polígono sem cruzamentos)."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def classificar_forma(pontas):
    """pontas: array (N, 2) com as pontas dos dedos levantados. Retorna (chave, polígono)."""
    n = len(pontas)
    if n < 2:
        return None, None
    if n == 2:
        a, b = pontas
        c = (a + b) / 2.0
        r = max(_dist(a, b) / 2, 1.0)
        ang = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        poly = np.stack([c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)], axis=1)
        return "circulo", poly.astype(np.float32)

    poly = ordenar_poligono(pontas)
    if n == 4:
        (_, _), (rw, rh), _ = cv2.minAreaRect(poly.astype(np.float32))
        if min(rw, rh) < 1:
            return None, poly
        if max(rw, rh) / min(rw, rh) < SQUARE_RATIO:
            return "quadrado", poly
        _, _, bw, bh = cv2.boundingRect(poly.astype(np.int32))
        return ("horizontal" if bw >= bh else "vertical"), poly
    return FORMA_POR_LADOS.get(n), poly


def aplicar_na_regiao(frame, poly, filtro, opacidade=1.0):
    """Aplica o filtro dentro do polígono, com borda suave e opacidade (para o fade)."""
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(poly.astype(np.int32))
    x0, y0, x1, y1 = max(x - 2, 0), max(y - 2, 0), min(x + bw + 2, w), min(y + bh + 2, h)
    if getattr(filtro, "alinhado", False):
        x0, y0 = (x0 // GRADE) * GRADE, (y0 // GRADE) * GRADE
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    roi = frame[y0:y1, x0:x1]
    filtrado = filtro(roi.copy())

    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [np.round((poly - [x0, y0]) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    peso = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32) * (opacidade / 255.0)
    roi[:] = cv2.blendLinear(filtrado, roi, peso, 1.0 - peso)


# ----------------------------------------------------------------------------
# Desenho: linhas finas com precisão sub-pixel e transparência
# ----------------------------------------------------------------------------

def _p(pt):
    """Ponto em coordenadas sub-pixel (shift=4) para linhas que deslizam sem 'pular' pixels."""
    return (int(round(pt[0] * 16)), int(round(pt[1] * 16)))


def camada(frame, pts, margem, opacidade, desenhar):
    """Desenha numa cópia da região em volta de pts e mistura com transparência."""
    h, w = frame.shape[:2]
    x, y, bw, bh = cv2.boundingRect(np.asarray(pts, np.float32).astype(np.int32))
    x0, y0 = max(x - margem, 0), max(y - margem, 0)
    x1, y1 = min(x + bw + margem, w), min(y + bh + margem, h)
    if x1 <= x0 or y1 <= y0:
        return
    roi = frame[y0:y1, x0:x1]
    copia = roi.copy()
    desenhar(copia, np.array([x0, y0], np.float32))
    cv2.addWeighted(copia, opacidade, roi, 1 - opacidade, 0, dst=roi)


def desenhar_mao(frame, mao, sprites):
    pts = mao.pts

    def ossos(img, o):
        for a, b in HAND_CONNECTIONS:
            cv2.line(img, _p(pts[a] - o), _p(pts[b] - o), (255, 255, 255), 1, cv2.LINE_AA, 4)
        for i, p in enumerate(pts):
            if i not in (4, 8, 12, 16, 20):
                cv2.circle(img, _p(p - o), 2 * 16, (255, 255, 255), -1, cv2.LINE_AA, 4)

    camada(frame, pts, 6, 0.45, ossos)

    for i, (nome, ponta, _) in enumerate(DEDOS):
        c = _p(pts[ponta])
        if mao.estados[i]:
            cv2.circle(frame, c, int(4.5 * 16), COR_DEDO[i], -1, cv2.LINE_AA, 4)
            cv2.circle(frame, c, 9 * 16, COR_DEDO[i], 1, cv2.LINE_AA, 4)
            sprites.append((texto_sprite(nome, 15, COR_DEDO[i], "semibold"),
                            pts[ponta][0] + 14, pts[ponta][1] - 26))
        else:
            cv2.circle(frame, c, int(2.5 * 16), (170, 170, 170), -1, cv2.LINE_AA, 4)
    sprites.append((texto_sprite(f"Mão {mao.lado}", 14, (235, 235, 235), "light"),
                    pts[0][0] - 40, pts[0][1] + 14))


def desenhar_conector(frame, poly, cor, opacidade):
    def linha(espessura):
        return lambda img, o: cv2.polylines(img, [np.round((poly - o) * 16).astype(np.int32)],
                                            True, cor, espessura, cv2.LINE_AA, 4)

    camada(frame, poly, 10, 0.16 * opacidade, linha(6))  # brilho suave por baixo
    camada(frame, poly, 4, 0.95 * opacidade, linha(1))   # linha fina por cima


# ----------------------------------------------------------------------------
# Sprites RGBA (texto e painel) desenhados com Pillow e colados com alfa
# ----------------------------------------------------------------------------

_FONTES = {
    "light": ["segoeuil.ttf", "seguisli.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "regular": ["segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"],
    "semibold": ["seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"],
}


@lru_cache(maxsize=None)
def fonte(tamanho, peso="regular"):
    for nome in _FONTES[peso]:
        for pasta in ("", "C:/Windows/Fonts/"):
            try:
                return ImageFont.truetype(pasta + nome, tamanho)
            except OSError:
                pass
    return ImageFont.load_default(size=tamanho)


def rgba(cor_bgr, alfa=255):
    return (cor_bgr[2], cor_bgr[1], cor_bgr[0], alfa)


def para_sprite(img_pil, escala):
    """RGBA do Pillow -> (cor pré-multiplicada BGR, 1 - alfa), ambos uint8 com 3 canais,
    reduzidos por 'escala' (super-amostragem)."""
    arr = np.asarray(img_pil).astype(np.float32) / 255.0
    alfa = np.ascontiguousarray(np.repeat(arr[..., 3:4], 3, axis=2))
    prem = np.ascontiguousarray(arr[..., 2::-1]) * alfa
    if escala != 1:
        w, h = img_pil.width // escala, img_pil.height // escala
        prem = cv2.resize(prem, (w, h), interpolation=cv2.INTER_AREA)
        alfa = cv2.resize(alfa, (w, h), interpolation=cv2.INTER_AREA)
    return (np.round(prem * 255.0).astype(np.uint8),
            np.round((1.0 - alfa) * 255.0).astype(np.uint8))


@lru_cache(maxsize=256)
def texto_sprite(txt, tam, cor_bgr, peso="regular", sombra=True):
    f = fonte(tam * 2, peso)
    l, t, r, b = f.getbbox(txt)
    pad = 12
    img = Image.new("RGBA", (r - l + 2 * pad, b - t + 2 * pad), (0, 0, 0, 0))
    if sombra:
        s = Image.new("RGBA", img.size, (0, 0, 0, 0))
        ImageDraw.Draw(s).text((pad - l, pad - t + 2), txt, font=f, fill=(0, 0, 0, 170))
        img = Image.alpha_composite(img, s.filter(ImageFilter.GaussianBlur(4)))
    ImageDraw.Draw(img).text((pad - l, pad - t), txt, font=f, fill=rgba(cor_bgr))
    return para_sprite(img, 2)


def colar(dst, sprite, x, y):
    """Cola um sprite com alfa em dst (operações do OpenCV, rápidas em Full HD)."""
    prem, inv_alfa = sprite
    x, y = int(round(x)), int(round(y))
    h, w = inv_alfa.shape[:2]
    H, W = dst.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    p = prem[y0 - y:y1 - y, x0 - x:x1 - x]
    ia = inv_alfa[y0 - y:y1 - y, x0 - x:x1 - x]
    reg = dst[y0:y1, x0:x1]
    tmp = cv2.multiply(reg, ia, scale=1.0 / 255.0)
    reg[:] = cv2.add(tmp, p)


# ----------------------------------------------------------------------------
# Painel lateral (vidro fosco)
# ----------------------------------------------------------------------------

PAINEL_A = ALTURA - 2 * MARGEM
LISTA_Y, LINHA_A = 118, 60
RODAPE_Y = LISTA_Y + LINHA_A * len(FORMAS) + 28


def pontos_icone(chave, cx, cy, r):
    """Vértices do ícone de cada forma (None para o círculo)."""
    if chave == "circulo":
        return None
    if chave == "quadrado":
        q = r * 0.8
        return [(cx - q, cy - q), (cx + q, cy - q), (cx + q, cy + q), (cx - q, cy + q)]
    if chave == "horizontal":
        return [(cx - r, cy - r * .55), (cx + r, cy - r * .55), (cx + r, cy + r * .55), (cx - r, cy + r * .55)]
    if chave == "vertical":
        return [(cx - r * .55, cy - r), (cx + r * .55, cy - r), (cx + r * .55, cy + r), (cx - r * .55, cy + r)]
    n = LADOS[chave]
    return [(cx + r * math.cos(-math.pi / 2 + k * 2 * math.pi / n),
             cy + r * math.sin(-math.pi / 2 + k * 2 * math.pi / n)) for k in range(n)]


def _texto(d, x, y, txt, tam, cor, peso="regular", anchor="la"):
    d.text((x * SS, y * SS), txt, font=fonte(tam * SS, peso), fill=cor, anchor=anchor)


@lru_cache(maxsize=1)
def painel_fixo():
    """Tudo que não muda no painel: borda, títulos, lista de formas e dicas."""
    W, H = PAINEL_L, PAINEL_A
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((0, 0, W * SS - 1, H * SS - 1), radius=20 * SS,
                        outline=(255, 255, 255, 38), width=SS)

    _texto(d, 28, 30, "Formas & filtros", 26, (255, 255, 255, 250), "semibold")
    _texto(d, 28, 70, "Levante os dedos para desenhar uma forma", 14, (255, 255, 255, 140), "light")

    for i, (chave, nome, como, nome_filtro, _, cor) in enumerate(FORMAS):
        y = LISTA_Y + i * LINHA_A
        cy = y + LINHA_A / 2
        pts = pontos_icone(chave, 44, cy, 13)
        if pts is None:
            d.ellipse(((44 - 13) * SS, (cy - 13) * SS, (44 + 13) * SS, (cy + 13) * SS),
                      outline=rgba(cor), width=int(1.3 * SS))
        else:
            d.polygon([(x * SS, yy * SS) for x, yy in pts], outline=rgba(cor), width=int(1.3 * SS))
        _texto(d, 76, y + 10, nome, 17, (255, 255, 255, 235), "regular")
        _texto(d, 76, y + 34, como, 13, (255, 255, 255, 120), "light")
        _texto(d, W - 26, y + 12, nome_filtro, 15, rgba(cor), "semibold", anchor="ra")
        if i < len(FORMAS) - 1:
            d.line((76 * SS, (y + LINHA_A) * SS, (W - 26) * SS, (y + LINHA_A) * SS),
                   fill=(255, 255, 255, 18), width=1)

    d.line((28 * SS, (RODAPE_Y - 14) * SS, (W - 28) * SS, (RODAPE_Y - 14) * SS),
           fill=(255, 255, 255, 40), width=1)
    _texto(d, 28, RODAPE_Y, "Dedos levantados", 14, (255, 255, 255, 150), "light")
    _texto(d, 28, H - 40, "F  tela cheia      H  painel      Q  sair", 13, (255, 255, 255, 110), "light")
    return para_sprite(img, SS)


@lru_cache(maxsize=64)
def painel_rodape(total, maos):
    """Parte de baixo do painel: total de dedos e os dedos de cada mão."""
    W, H = PAINEL_L, 170
    img = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    _texto(d, W - 28, -8, str(total), 40, (255, 255, 255, 240), "light", anchor="ra")
    if not maos:
        _texto(d, 28, 44, "Nenhuma mão na câmera", 14, (255, 255, 255, 110), "light")
    for j, (lado, estados) in enumerate(maos[:2]):
        y = 44 + j * 58
        _texto(d, 28, y, f"Mão {lado}", 14, (255, 255, 255, 200), "regular")
        for k, curto in enumerate(DEDOS_CURTO):
            x = 30 + k * 74
            cor = rgba(COR_DEDO[k]) if estados[k] else (255, 255, 255, 60)
            r = 4
            cy = y + 34
            if estados[k]:
                d.ellipse(((x - r) * SS, (cy - r) * SS, (x + r) * SS, (cy + r) * SS), fill=cor)
            else:
                d.ellipse(((x - r) * SS, (cy - r) * SS, (x + r) * SS, (cy + r) * SS), outline=cor, width=SS)
            _texto(d, x + 11, cy, curto, 13, cor, "regular", anchor="lm")
    return para_sprite(img, SS)


@lru_cache(maxsize=4)
def mascara_arredondada(w, h, raio):
    img = Image.new("L", (w * SS, h * SS), 0)
    ImageDraw.Draw(img).rounded_rectangle((0, 0, w * SS - 1, h * SS - 1), radius=raio * SS, fill=255)
    m = np.asarray(img.resize((w, h), Image.LANCZOS)).astype(np.float32) / 255.0
    return m


def desenhar_painel(canvas, px, destaque_y, forma_ativa, rodape):
    W, H = PAINEL_L, PAINEL_A
    py = MARGEM
    vis = min(W, canvas.shape[1] - px)
    if vis <= 0:
        return
    painel = np.zeros((H, W, 3), np.uint8)
    painel[:, :vis] = canvas[py:py + H, px:px + vis]

    # vidro fosco: desfoca o fundo em baixa resolução e escurece
    fosco = cv2.resize(painel, (W // 10, H // 10), interpolation=cv2.INTER_AREA)
    fosco = cv2.GaussianBlur(fosco, (0, 0), 2.0)
    fosco = cv2.addWeighted(fosco, 0.35, np.full_like(fosco, (26, 22, 20)), 0.65, 0)
    fosco = cv2.resize(fosco, (W, H), interpolation=cv2.INTER_LINEAR)

    # destaque da forma ativa (desliza suavemente entre as linhas)
    if forma_ativa is not None and destaque_y is not None:
        cor = FORMA[forma_ativa][5]
        y0, y1 = int(destaque_y) + 4, int(destaque_y) + LINHA_A - 4
        faixa = fosco[y0:y1, 12:W - 12]
        cv2.addWeighted(faixa, 0.82, np.full_like(faixa, cor), 0.18, 0, dst=faixa)
        cv2.line(fosco, (14 * 16, y0 * 16 + 8 * 16), (14 * 16, y1 * 16 - 8 * 16), cor, 2, cv2.LINE_AA, 4)
        i = INDICE_FORMA[forma_ativa]
        cy = LISTA_Y + i * LINHA_A + LINHA_A / 2
        pts = pontos_icone(forma_ativa, 44, cy, 13)
        if pts is None:
            cv2.circle(fosco, _p((44, cy)), 13 * 16, cor, -1, cv2.LINE_AA, 4)
        else:
            cv2.fillPoly(fosco, [np.array([_p(p) for p in pts], np.int32)], cor, cv2.LINE_AA, 4)

    colar(fosco, painel_fixo(), 0, 0)
    colar(fosco, rodape, 0, RODAPE_Y)

    m = mascara_arredondada(W, H, 20)
    painel = cv2.blendLinear(fosco, painel, m, 1.0 - m)
    canvas[py:py + H, px:px + vis] = painel[:, :vis]


# ----------------------------------------------------------------------------
# Câmera em thread e detector assíncrono
# ----------------------------------------------------------------------------

class Camera:
    """Lê a webcam numa thread separada e guarda sempre o quadro mais recente."""

    def __init__(self, indice=0):
        backend = cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY
        self.cap = cv2.VideoCapture(indice, backend)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
            raise SystemExit("Não foi possível abrir a webcam.")
        self.frame, self.n, self.rodando = None, 0, True
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.rodando:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                self.frame, self.n = frame, self.n + 1
                self.cond.notify_all()

    def ler(self, ultimo):
        """Espera um quadro mais novo que 'ultimo'."""
        with self.cond:
            self.cond.wait_for(lambda: self.n != ultimo or not self.rodando, timeout=1.0)
            return self.n, self.frame

    def fechar(self):
        self.rodando = False
        self.thread.join(timeout=1)
        self.cap.release()


def garantir_modelo():
    if not os.path.exists(MODEL_PATH):
        print("Baixando modelo de mãos do MediaPipe...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Modelo salvo em", MODEL_PATH)


class Detector:
    """HandLandmarker em modo LIVE_STREAM: detect_async não bloqueia o desenho."""

    def __init__(self):
        garantir_modelo()
        self.resultado = None  # (timestamp_ms, result)
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            result_callback=self._callback,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.ultimo_ts = -1

    def _callback(self, result, _imagem, ts):
        self.resultado = (ts, result)

    def enviar(self, frame_bgr, ts):
        if ts <= self.ultimo_ts:
            ts = self.ultimo_ts + 1
        self.ultimo_ts = ts
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (DETECCAO_L, int(DETECCAO_L * h / w)), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        self.landmarker.detect_async(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)

    def fechar(self):
        self.landmarker.close()


def tamanho_tela():
    try:
        import ctypes
        u = ctypes.windll.user32
        u.SetProcessDPIAware()  # janela em pixels reais (sem borrão do zoom do Windows)
        return u.GetSystemMetrics(0), u.GetSystemMetrics(1)
    except Exception:
        return LARGURA, ALTURA


def suavizar(atual, alvo, dt, velocidade=14.0):
    """Aproxima 'atual' de 'alvo' de forma exponencial, independente do FPS."""
    return alvo if atual is None else atual + (alvo - atual) * (1 - math.exp(-velocidade * dt))


# ----------------------------------------------------------------------------
# Principal
# ----------------------------------------------------------------------------

def main():
    tela_w, tela_h = tamanho_tela()
    detector = Detector()
    cam = Camera(0)

    janela = "Filtro com as maos"
    cv2.namedWindow(janela, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    escala = min(1.0, 0.92 * tela_w / LARGURA, 0.86 * tela_h / ALTURA)
    cv2.resizeWindow(janela, int(LARGURA * escala), int(ALTURA * escala))
    tela_cheia = False

    painel_fixo()  # desenha o painel uma vez antes de começar

    maos = []
    ultimo_resultado = None
    forma_atual, candidata, contagem = None, None, 0
    fade_inicio = 0.0
    destaque_y = None
    painel_x = float(LARGURA - PAINEL_L - MARGEM)
    mostrar_painel = True
    n = 0
    t0 = time.monotonic()
    t_ant = t0
    fps = 30.0

    try:
        while True:
            n, frame = cam.ler(n)
            if frame is None:
                continue
            agora = time.monotonic()
            dt = max(agora - t_ant, 1e-3)
            t_ant = agora

            frame = cv2.flip(frame, 1)  # espelho
            if frame.shape[1] != LARGURA or frame.shape[0] != ALTURA:
                frame = cv2.resize(frame, (LARGURA, ALTURA), interpolation=cv2.INTER_LINEAR)
            detector.enviar(frame, int((agora - t0) * 1000))

            # novo resultado do detector -> atualiza as mãos
            res = detector.resultado
            if res is not None and res is not ultimo_resultado:
                ultimo_resultado = res
                ts, result = res
                deteccoes = []
                for lms, hd in zip(result.hand_landmarks, result.handedness):
                    pts = np.array([[lm.x * LARGURA, lm.y * ALTURA] for lm in lms], np.float32)
                    lado = "direita" if hd[0].category_name == "Right" else "esquerda"
                    deteccoes.append((pts, lado))
                deteccoes.sort(key=lambda d: d[0][0, 0])  # da esquerda para a direita
                if len(deteccoes) != len(maos):
                    maos = [Mao() for _ in deteccoes]
                for mao, (pts, lado) in zip(maos, deteccoes):
                    mao.atualizar(pts, lado, ts / 1000.0)

            pontas = [m.pts[p] for m in maos for (_, p, _), e in zip(DEDOS, m.estados) if e]
            pontas = np.array(pontas, np.float32).reshape(-1, 2)
            forma, poly = classificar_forma(pontas)

            # só troca de filtro depois que a forma fica estável por alguns quadros
            if forma is None:
                forma_atual, candidata, contagem = None, None, 0
            elif forma != forma_atual:
                if forma == candidata:
                    contagem += 1
                else:
                    candidata, contagem = forma, 1
                if contagem >= SHAPE_STABLE_FRAMES:
                    forma_atual, candidata, contagem = forma, None, 0
                    fade_inicio = agora
            else:
                candidata, contagem = None, 0

            sprites = []
            if forma_atual is not None and poly is not None:
                _, nome, _, nome_filtro, filtro, cor = FORMA[forma_atual]
                k = min(1.0, (agora - fade_inicio) / FADE)
                opac = k * k * (3 - 2 * k)  # smoothstep
                aplicar_na_regiao(frame, poly, filtro, opac)
                desenhar_conector(frame, poly, cor, opac)
                sprites.append((texto_sprite(f"{nome}  ·  {nome_filtro}", 20, cor, "semibold"),
                                poly[:, 0].min() - 6, poly[:, 1].min() - 50))
            for mao in maos:
                desenhar_mao(frame, mao, sprites)
            if not maos:
                s = texto_sprite("Mostre as mãos para a câmera", 22, (255, 255, 255), "light")
                sprites.append((s, (LARGURA - PAINEL_L - s[1].shape[1]) / 2, ALTURA - 90))

            fps = 0.92 * fps + 0.08 / dt
            sprites.append((texto_sprite(f"{fps:.0f} fps", 13, (255, 255, 255), "light"), 22, 18))
            for spr, x, y in sprites:
                colar(frame, spr, x, y)

            # painel com animação de entrada/saída e destaque deslizante
            alvo_x = LARGURA - PAINEL_L - MARGEM if mostrar_painel else LARGURA + 4
            painel_x = suavizar(painel_x, alvo_x, dt, 12.0)
            if forma_atual is not None:
                destaque_y = suavizar(destaque_y, LISTA_Y + INDICE_FORMA[forma_atual] * LINHA_A, dt, 16.0)
            rodape = painel_rodape(len(pontas), tuple((m.lado, tuple(m.estados)) for m in maos))
            desenhar_painel(frame, int(round(painel_x)), destaque_y, forma_atual, rodape)

            cv2.imshow(janela, frame)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord("h"):
                mostrar_painel = not mostrar_painel
            if tecla == ord("f"):
                tela_cheia = not tela_cheia
                cv2.setWindowProperty(janela, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_FULLSCREEN if tela_cheia else cv2.WINDOW_NORMAL)
            if cv2.getWindowProperty(janela, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cam.fechar()
        detector.fechar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
