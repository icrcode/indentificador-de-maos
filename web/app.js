// Filtro com as mãos — versão para navegador
// Detecta as mãos com o MediaPipe, liga as pontas dos dedos levantados formando
// uma figura e aplica, só dentro dela, um filtro (WebGL) que depende da forma.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/vision_bundle.mjs";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.0.1/wasm";
const MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";

// ---------------------------------------------------------------------------
// Configuração
// ---------------------------------------------------------------------------

const REF_ALTURA = 1080;          // espaço de referência para suavização (independe da tela)
const MIN_CUTOFF = 1.4;           // One Euro: menos tremida parado
const BETA = 0.008;               // One Euro: menos atraso em movimento
const SHAPE_STABLE_FRAMES = 4;    // quadros seguidos para trocar de forma
const FADE = 0.25;                // segundos de transição do filtro
const SQUARE_RATIO = 1.35;        // até esta proporção, 4 pontos são "quadrado"
const DEDO_SOBE = 1.2, DEDO_DESCE = 1.08;
const POLEGAR_SOBE = 0.55, POLEGAR_DESCE = 0.42;

// [nome, ponta, articulação do meio]
const DEDOS = [["Polegar", 4, 3], ["Indicador", 8, 6], ["Médio", 12, 10], ["Anelar", 16, 14], ["Mínimo", 20, 18]];
const DEDOS_CURTO = ["Pol", "Ind", "Méd", "Anel", "Mín"];
const COR_DEDO = ["#fb7185", "#fdba74", "#fde047", "#86efac", "#93c5fd"];
const PONTAS = new Set([4, 8, 12, 16, 20]);

const CONEXOES = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];

// chave, nome, como fazer, filtro, modo do shader, cor
const FORMAS = [
  ["circulo", "Círculo", "2 dedos · polegar + indicador", "Lupa", 1, "#7dd3fc"],
  ["triangulo", "Triângulo", "3 dedos", "Pontilhado P&B", 2, "#f5f5f4"],
  ["quadrado", "Quadrado", "4 dedos · moldura quadrada", "Pontilhado cor", 3, "#fbbf24"],
  ["horizontal", "Retângulo deitado", "4 dedos · moldura larga", "Cartoon", 4, "#86efac"],
  ["vertical", "Retângulo em pé", "4 dedos · moldura alta", "Azul", 5, "#60a5fa"],
  ["pentagono", "Pentágono", "5 dedos · uma mão aberta", "Lápis", 6, "#d6d3d1"],
  ["hexagono", "Hexágono", "6 dedos · 3 em cada mão", "Térmico", 7, "#fb923c"],
  ["heptagono", "Heptágono", "7 dedos · mão aberta + 2", "Pixelado", 8, "#e879f9"],
  ["octogono", "Octógono", "8 dedos · 4 em cada mão", "Negativo", 9, "#f9a8d4"],
  ["eneagono", "Eneágono", "9 dedos", "Sépia", 10, "#d4a373"],
  ["decagono", "Decágono", "10 dedos · duas mãos abertas", "Neon", 11, "#a78bfa"],
].map(([chave, nome, como, filtro, modo, cor]) => ({ chave, nome, como, filtro, modo, cor }));
const FORMA = Object.fromEntries(FORMAS.map((f) => [f.chave, f]));
const LADOS = { triangulo: 3, pentagono: 5, hexagono: 6, heptagono: 7, octogono: 8, eneagono: 9, decagono: 10 };
const FORMA_POR_LADOS = Object.fromEntries(Object.entries(LADOS).map(([k, n]) => [n, k]));

// ---------------------------------------------------------------------------
// Suavização One Euro e estado dos dedos
// ---------------------------------------------------------------------------

class OneEuro {
  constructor() { this.x = null; }

  static alpha(dt, cutoff) {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  // pts: array de [x, y]
  filtrar(pts, t) {
    if (!this.x) {
      this.x = pts.map((p) => [...p]);
      this.dx = pts.map(() => [0, 0]);
      this.t = t;
      return this.x;
    }
    const dt = Math.max(t - this.t, 1e-3);
    const aD = OneEuro.alpha(dt, 1.0);
    for (let i = 0; i < pts.length; i++) {
      const [x, y] = pts[i];
      const [px, py] = this.x[i];
      const dx = aD * ((x - px) / dt) + (1 - aD) * this.dx[i][0];
      const dy = aD * ((y - py) / dt) + (1 - aD) * this.dx[i][1];
      this.dx[i] = [dx, dy];
      const a = OneEuro.alpha(dt, MIN_CUTOFF + BETA * Math.hypot(dx, dy));
      this.x[i] = [a * x + (1 - a) * px, a * y + (1 - a) * py];
    }
    this.t = t;
    return this.x;
  }
}

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function dedosLevantados(p, antes) {
  const palma = Math.max(dist(p[0], p[9]), 1e-6);
  const s = dist(p[4], p[5]) / palma;
  const r = dist(p[4], p[17]) / Math.max(dist(p[3], p[17]), 1e-6);
  const estados = [antes[0] ? s > POLEGAR_DESCE && r > 1.0 : s > POLEGAR_SOBE && r > 1.05];
  for (let i = 1; i < 5; i++) {
    const [, ponta, meio] = DEDOS[i];
    const razao = dist(p[ponta], p[0]) / Math.max(dist(p[meio], p[0]), 1e-6);
    estados.push(razao > (antes[i] ? DEDO_DESCE : DEDO_SOBE));
  }
  return estados;
}

class Mao {
  constructor() {
    this.suave = new OneEuro();
    this.ref = null;       // pontos suavizados no espaço de referência
    this.lado = "";
    this.estados = [false, false, false, false, false];
  }

  atualizar(ref, lado, t) {
    this.ref = this.suave.filtrar(ref, t);
    this.lado = lado;
    this.estados = dedosLevantados(this.ref, this.estados);
  }
}

// ---------------------------------------------------------------------------
// Geometria
// ---------------------------------------------------------------------------

function ordenarPoligono(pts) {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return [...pts].sort((a, b) => Math.atan2(a[1] - cy, a[0] - cx) - Math.atan2(b[1] - cy, b[0] - cx));
}

function envoltoria(pts) {
  const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cruz = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const baixo = [], cima = [];
  for (const q of p) {
    while (baixo.length >= 2 && cruz(baixo.at(-2), baixo.at(-1), q) <= 0) baixo.pop();
    baixo.push(q);
  }
  for (const q of p.reverse()) {
    while (cima.length >= 2 && cruz(cima.at(-2), cima.at(-1), q) <= 0) cima.pop();
    cima.push(q);
  }
  return baixo.slice(0, -1).concat(cima.slice(0, -1));
}

// Lados do menor retângulo (qualquer rotação) que contém os pontos
function menorRetangulo(pts) {
  const h = envoltoria(pts);
  let melhor = null;
  for (let i = 0; i < h.length; i++) {
    const a = h[i], b = h[(i + 1) % h.length];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
    const ux = (b[0] - a[0]) / len, uy = (b[1] - a[1]) / len;
    let min1 = Infinity, max1 = -Infinity, min2 = Infinity, max2 = -Infinity;
    for (const q of h) {
      const d1 = q[0] * ux + q[1] * uy, d2 = -q[0] * uy + q[1] * ux;
      min1 = Math.min(min1, d1); max1 = Math.max(max1, d1);
      min2 = Math.min(min2, d2); max2 = Math.max(max2, d2);
    }
    const w = max1 - min1, hh = max2 - min2;
    if (!melhor || w * hh < melhor[0] * melhor[1]) melhor = [w, hh];
  }
  return melhor || [0, 0];
}

function caixa(pts) {
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (const [x, y] of pts) {
    x0 = Math.min(x0, x); y0 = Math.min(y0, y);
    x1 = Math.max(x1, x); y1 = Math.max(y1, y);
  }
  return { x0, y0, x1, y1, w: x1 - x0, h: y1 - y0 };
}

// pontas: [[x, y], ...] em pixels de tela -> { chave, poly }
function classificarForma(pontas) {
  const n = pontas.length;
  if (n < 2) return { chave: null, poly: null };
  if (n === 2) {
    const [a, b] = pontas;
    const c = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
    const r = Math.max(dist(a, b) / 2, 1);
    const poly = [];
    for (let k = 0; k < 72; k++) {
      const t = (k / 72) * Math.PI * 2;
      poly.push([c[0] + r * Math.cos(t), c[1] + r * Math.sin(t)]);
    }
    return { chave: "circulo", poly };
  }
  const poly = ordenarPoligono(pontas);
  if (n === 4) {
    const [rw, rh] = menorRetangulo(poly);
    if (Math.min(rw, rh) < 1) return { chave: null, poly };
    if (Math.max(rw, rh) / Math.min(rw, rh) < SQUARE_RATIO) return { chave: "quadrado", poly };
    const bb = caixa(poly);
    return { chave: bb.w >= bb.h ? "horizontal" : "vertical", poly };
  }
  return { chave: FORMA_POR_LADOS[n] ?? null, poly };
}

// ---------------------------------------------------------------------------
// Filtros em WebGL (um shader, vários modos)
// ---------------------------------------------------------------------------

const VERT = `#version 300 es
in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG = `#version 300 es
precision highp float;
uniform sampler2D uVideo;
uniform vec2 uRes;      // tamanho da tela em pixels
uniform vec2 uOff;      // posição do vídeo (cover) na tela
uniform vec2 uSize;     // tamanho do vídeo (cover) na tela
uniform vec2 uCentro;   // centro da forma (lupa)
uniform float uS;       // escala dos padrões (densidade de pixels)
uniform int uModo;
out vec4 saida;

// pixel da tela (origem no topo, espelhado) -> cor do vídeo
vec3 px(vec2 p) {
  vec2 uv = vec2((uRes.x - p.x - uOff.x) / uSize.x, (p.y - uOff.y) / uSize.y);
  return texture(uVideo, clamp(uv, 0.0, 1.0)).rgb;
}
float lum(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float sobel(vec2 p, float d) {
  float tl = lum(px(p + vec2(-d, -d))), t = lum(px(p + vec2(0, -d))), tr = lum(px(p + vec2(d, -d)));
  float l  = lum(px(p + vec2(-d, 0))),                                 r  = lum(px(p + vec2(d, 0)));
  float bl = lum(px(p + vec2(-d, d))),  b = lum(px(p + vec2(0, d))),  br = lum(px(p + vec2(d, d)));
  float gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
  float gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;
  return length(vec2(gx, gy));
}

vec3 media4(vec2 c, float r) {
  return (px(c + vec2(-r, -r)) + px(c + vec2(r, -r)) + px(c + vec2(-r, r)) + px(c + vec2(r, r))) * 0.25;
}

vec3 hsv2rgb(vec3 c) {
  vec3 p = abs(fract(c.xxx + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0);
  return c.z * mix(vec3(1.0), clamp(p - 1.0, 0.0, 1.0), c.y);
}

vec3 inferno(float t) {
  const vec3 c0 = vec3(0.0002189403691192265, 0.001651004631001012, -0.01948089843709184);
  const vec3 c1 = vec3(0.1065134194856116, 0.5639564367884091, 3.932712388889277);
  const vec3 c2 = vec3(11.60249308247187, -3.972853965665698, -15.9423941062914);
  const vec3 c3 = vec3(-41.70399613139459, 17.43639888205313, 44.35414519872813);
  const vec3 c4 = vec3(77.162935699427, -33.40235894210092, -81.80730925738993);
  const vec3 c5 = vec3(-71.31942824499214, 32.62606426397723, 73.20951985803202);
  const vec3 c6 = vec3(25.13112622477341, -12.24266895238567, -23.07032500287172);
  return clamp(c0 + t * (c1 + t * (c2 + t * (c3 + t * (c4 + t * (c5 + t * c6))))), 0.0, 1.0);
}

void main() {
  vec2 p = vec2(gl_FragCoord.x, uRes.y - gl_FragCoord.y);
  vec3 c = px(p);
  vec3 o = c;

  if (uModo == 1) {                     // lupa
    o = px(uCentro + (p - uCentro) * 0.5);

  } else if (uModo == 2) {              // pontilhado preto e branco
    float cell = 8.0 * uS;
    vec2 cc = (floor(p / cell) + 0.5) * cell;
    float l = clamp((lum(media4(cc, cell * 0.25)) - 0.5) * 1.3 + 0.5, 0.0, 1.0);
    float r = sqrt(1.0 - l) * cell * 0.5 * 1.3;
    float a = clamp(r - length(p - cc) + 0.5, 0.0, 1.0);
    o = mix(vec3(0.95), vec3(0.08), a);

  } else if (uModo == 3) {              // pontilhado colorido
    float cell = 12.0 * uS;
    vec2 cc = (floor(p / cell) + 0.5) * cell;
    vec3 m = media4(cc, cell * 0.25);
    float l = lum(m);
    vec3 sat = clamp(mix(vec3(l), m, 1.6) * 1.15, 0.0, 1.0);
    float r = (0.2 + 0.62 * sqrt(l)) * cell * 0.6;
    float a = clamp(r - length(p - cc) + 0.5, 0.0, 1.0);
    o = mix(vec3(0.07), sat, a);

  } else if (uModo == 4) {              // cartoon
    float d = 2.0 * uS;
    vec3 s = vec3(0.0);
    for (int y = -1; y <= 1; y++)
      for (int x = -1; x <= 1; x++) s += px(p + vec2(x, y) * d);
    s /= 9.0;
    vec3 q = floor(s * 6.0) / 6.0 + 1.0 / 12.0;
    float e = smoothstep(0.22, 0.42, sobel(p, 1.5 * uS));
    o = mix(q, vec3(0.03), e);

  } else if (uModo == 5) {              // azul
    float g = lum(c);
    o = clamp(vec3(g * 0.35, g * 0.75 + 0.04, g * 1.1 + 0.235), 0.0, 1.0);

  } else if (uModo == 6) {              // lápis (color dodge)
    float g = lum(c);
    float soma = 0.0;
    for (int k = 0; k < 12; k++) {
      float t = float(k) * 0.5235988;
      vec2 dir = vec2(cos(t), sin(t));
      soma += lum(px(p + dir * 3.0 * uS)) + lum(px(p + dir * 7.0 * uS));
    }
    float blur = soma / 24.0;
    o = vec3(clamp(g / max(blur, 1e-3), 0.0, 1.0));

  } else if (uModo == 7) {              // térmico
    o = inferno(clamp((lum(c) - 0.08) * 1.25, 0.0, 1.0));

  } else if (uModo == 8) {              // pixelado
    float b = 24.0 * uS;
    vec2 cc = (floor(p / b) + 0.5) * b;
    o = media4(cc, b * 0.25);

  } else if (uModo == 9) {              // negativo
    o = 1.0 - c;

  } else if (uModo == 10) {             // sépia
    o = clamp(vec3(dot(c, vec3(0.393, 0.769, 0.189)),
                   dot(c, vec3(0.349, 0.686, 0.168)),
                   dot(c, vec3(0.272, 0.534, 0.131))), 0.0, 1.0);

  } else if (uModo == 11) {             // neon
    vec3 cor = hsv2rgb(vec3(p.x / uRes.x, 0.85, 1.0));
    float e = smoothstep(0.18, 0.45, sobel(p, 1.0 * uS));
    float g = smoothstep(0.1, 0.6, sobel(p, 4.0 * uS)) * 0.45 + smoothstep(0.1, 0.6, sobel(p, 9.0 * uS)) * 0.25;
    o = cor * clamp(e + g, 0.0, 1.2);
  }
  saida = vec4(o, 1.0);
}`;

class Filtros {
  constructor() {
    this.canvas = document.createElement("canvas");
    const gl = this.canvas.getContext("webgl2", { premultipliedAlpha: false, preserveDrawingBuffer: true, antialias: false });
    if (!gl) throw new Error("Seu navegador não suporta WebGL2.");
    this.gl = gl;

    const compilar = (tipo, src) => {
      const s = gl.createShader(tipo);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const prog = gl.createProgram();
    gl.attachShader(prog, compilar(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, compilar(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    this.u = {};
    for (const n of ["uRes", "uOff", "uSize", "uCentro", "uS", "uModo", "uVideo"]) this.u[n] = gl.getUniformLocation(prog, n);

    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
    const loc = gl.getAttribLocation(prog, "aPos");
    gl.enableVertexAttribArray(loc);
    gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

    this.tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.uniform1i(this.u.uVideo, 0);
  }

  // Renderiza o filtro só na caixa (bb) da forma
  render(video, W, H, cover, modo, centro, escala, bb) {
    const gl = this.gl;
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W;
      this.canvas.height = H;
    }
    gl.viewport(0, 0, W, H);
    gl.bindTexture(gl.TEXTURE_2D, this.tex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
    gl.uniform2f(this.u.uRes, W, H);
    gl.uniform2f(this.u.uOff, cover.ox, cover.oy);
    gl.uniform2f(this.u.uSize, cover.dw, cover.dh);
    gl.uniform2f(this.u.uCentro, centro[0], centro[1]);
    gl.uniform1f(this.u.uS, escala);
    gl.uniform1i(this.u.uModo, modo);
    const x0 = Math.max(0, Math.floor(bb.x0) - 4), x1 = Math.min(W, Math.ceil(bb.x1) + 4);
    const y0 = Math.max(0, Math.floor(bb.y0) - 4), y1 = Math.min(H, Math.ceil(bb.y1) + 4);
    gl.enable(gl.SCISSOR_TEST);
    gl.scissor(x0, H - y1, Math.max(0, x1 - x0), Math.max(0, y1 - y0));
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return this.canvas;
  }
}

// ---------------------------------------------------------------------------
// Painel (HTML)
// ---------------------------------------------------------------------------

function iconeSVG(chave) {
  const c = 15, r = 12;
  let forma;
  if (chave === "circulo") forma = `<circle class="forma" cx="${c}" cy="${c}" r="${r}"/>`;
  else {
    let pts;
    if (chave === "quadrado") { const q = r * 0.8; pts = [[c - q, c - q], [c + q, c - q], [c + q, c + q], [c - q, c + q]]; }
    else if (chave === "horizontal") pts = [[c - r, c - r * 0.55], [c + r, c - r * 0.55], [c + r, c + r * 0.55], [c - r, c + r * 0.55]];
    else if (chave === "vertical") pts = [[c - r * 0.55, c - r], [c + r * 0.55, c - r], [c + r * 0.55, c + r], [c - r * 0.55, c + r]];
    else {
      const n = LADOS[chave];
      pts = Array.from({ length: n }, (_, k) => {
        const a = -Math.PI / 2 + (k * 2 * Math.PI) / n;
        return [c + r * Math.cos(a), c + r * Math.sin(a)];
      });
    }
    forma = `<polygon class="forma" stroke-linejoin="round" points="${pts.map((p) => p.map((v) => v.toFixed(2)).join(",")).join(" ")}"/>`;
  }
  return `<svg viewBox="0 0 30 30" aria-hidden="true">${forma}</svg>`;
}

const ui = {
  painel: document.getElementById("painel"),
  lista: document.getElementById("lista"),
  destaque: document.getElementById("destaque"),
  total: document.getElementById("total"),
  maos: document.getElementById("maos"),
  btnMostrar: document.getElementById("btnMostrar"),
  inicio: document.getElementById("inicio"),
  btnIniciar: document.getElementById("btnIniciar"),
  status: document.getElementById("status"),
};

ui.lista.innerHTML = FORMAS.map((f) => `
  <li data-chave="${f.chave}" style="color:${f.cor}">
    ${iconeSVG(f.chave)}
    <div style="color:var(--texto)"><span class="nome">${f.nome}</span><span class="como">${f.como}</span></div>
    <span class="filtro">${f.filtro}</span>
  </li>`).join("");

let formaNoPainel;
function painelForma(chave) {
  if (chave === formaNoPainel) return;
  formaNoPainel = chave;
  for (const li of ui.lista.children) li.classList.toggle("ativo", li.dataset.chave === chave);
  moverDestaque();
}

function moverDestaque() {
  const li = formaNoPainel && ui.lista.querySelector(`[data-chave="${formaNoPainel}"]`);
  ui.destaque.classList.toggle("visivel", !!li);
  if (!li) return;
  ui.destaque.style.setProperty("--cor", FORMA[formaNoPainel].cor);
  ui.destaque.style.transform = `translateY(${li.offsetTop}px)`;
  ui.destaque.style.height = `${li.offsetHeight}px`;
}
window.addEventListener("resize", moverDestaque);

let rodapeNoPainel = "";
function painelRodape(total, maos) {
  const chave = total + "|" + maos.map((m) => m.lado + m.estados.join()).join("|");
  if (chave === rodapeNoPainel) return;
  rodapeNoPainel = chave;
  ui.total.textContent = total;
  if (!maos.length) {
    ui.maos.innerHTML = `<div class="vazio">Nenhuma mão na câmera</div>`;
    return;
  }
  ui.maos.innerHTML = maos.slice(0, 2).map((m) => `
    <div class="mao">
      <div class="titulo">Mão ${m.lado}</div>
      <div class="dedos">${DEDOS_CURTO.map((d, k) =>
        `<span class="dedo${m.estados[k] ? " on" : ""}" style="--cor:${COR_DEDO[k]}"><i></i>${d}</span>`).join("")}
      </div>
    </div>`).join("");
}

function alternarPainel() {
  const oculto = ui.painel.classList.toggle("oculto");
  ui.btnMostrar.classList.toggle("visivel", oculto);
}

function alternarTelaCheia() {
  if (document.fullscreenElement) document.exitFullscreen();
  else document.documentElement.requestFullscreen?.();
}

document.getElementById("btnTela").addEventListener("click", alternarTelaCheia);
document.getElementById("btnEsconder").addEventListener("click", alternarPainel);
ui.btnMostrar.addEventListener("click", alternarPainel);
window.addEventListener("keydown", (e) => {
  if (e.repeat || e.ctrlKey || e.metaKey || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "f") alternarTelaCheia();
  if (k === "h") alternarPainel();
});

painelRodape(0, []);

// ---------------------------------------------------------------------------
// Desenho no canvas
// ---------------------------------------------------------------------------

const tela = document.getElementById("tela");
const ctx = tela.getContext("2d");
const video = document.getElementById("video");
const FONTE = `"Inter", "Segoe UI", system-ui, sans-serif`;

function texto(txt, x, y, tam, cor, peso = 400, dpr = 1) {
  ctx.font = `${peso} ${tam * dpr}px ${FONTE}`;
  ctx.shadowColor = "rgba(0,0,0,0.65)";
  ctx.shadowBlur = 6 * dpr;
  ctx.shadowOffsetY = 1 * dpr;
  ctx.fillStyle = cor;
  ctx.fillText(txt, x, y);
  ctx.shadowColor = "transparent";
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;
}

function caminho(poly) {
  ctx.beginPath();
  ctx.moveTo(poly[0][0], poly[0][1]);
  for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
  ctx.closePath();
}

function desenharMao(pts, mao, dpr) {
  ctx.lineWidth = dpr;
  ctx.strokeStyle = "rgba(255,255,255,0.45)";
  ctx.beginPath();
  for (const [a, b] of CONEXOES) {
    ctx.moveTo(pts[a][0], pts[a][1]);
    ctx.lineTo(pts[b][0], pts[b][1]);
  }
  ctx.stroke();

  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.beginPath();
  pts.forEach((p, i) => {
    if (PONTAS.has(i)) return;
    ctx.moveTo(p[0] + 2 * dpr, p[1]);
    ctx.arc(p[0], p[1], 2 * dpr, 0, Math.PI * 2);
  });
  ctx.fill();

  DEDOS.forEach(([nome, ponta], i) => {
    const [x, y] = pts[ponta];
    ctx.beginPath();
    if (mao.estados[i]) {
      ctx.fillStyle = COR_DEDO[i];
      ctx.arc(x, y, 4.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
      ctx.beginPath();
      ctx.strokeStyle = COR_DEDO[i];
      ctx.lineWidth = dpr;
      ctx.arc(x, y, 9 * dpr, 0, Math.PI * 2);
      ctx.stroke();
      texto(nome, x + 14 * dpr, y - 12 * dpr, 14, COR_DEDO[i], 600, dpr);
    } else {
      ctx.fillStyle = "rgba(200,200,200,0.8)";
      ctx.arc(x, y, 2.5 * dpr, 0, Math.PI * 2);
      ctx.fill();
    }
  });
  texto(`Mão ${mao.lado}`, pts[0][0] - 36 * dpr, pts[0][1] + 28 * dpr, 13, "rgba(255,255,255,0.85)", 300, dpr);
}

// ---------------------------------------------------------------------------
// Modo demonstração: ?demo=hexagono mostra a forma sem precisar das mãos
// ---------------------------------------------------------------------------

const DEMO = FORMA[new URLSearchParams(location.search).get("demo")] ? new URLSearchParams(location.search).get("demo") : null;

function pontasDemo(chave, W, H, agora) {
  const cx = W * 0.4, cy = H * 0.5, r = Math.min(W, H) * 0.28;
  const giro = Math.sin(agora / 1500) * 0.15;
  if (chave === "circulo") return [[cx - r, cy], [cx + r, cy]];
  if (chave === "quadrado") return [[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r]];
  if (chave === "horizontal") return [[cx - r * 1.5, cy - r * 0.7], [cx + r * 1.5, cy - r * 0.7], [cx + r * 1.5, cy + r * 0.7], [cx - r * 1.5, cy + r * 0.7]];
  if (chave === "vertical") return [[cx - r * 0.6, cy - r], [cx + r * 0.6, cy - r], [cx + r * 0.6, cy + r], [cx - r * 0.6, cy + r]];
  const n = LADOS[chave];
  return Array.from({ length: n }, (_, k) => {
    const a = giro - Math.PI / 2 + (k * 2 * Math.PI) / n;
    return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
  });
}

// ---------------------------------------------------------------------------
// Loop principal
// ---------------------------------------------------------------------------

let landmarker, filtros;
let maos = [];
let formaAtual = null, candidata = null, contagem = 0, fadeInicio = 0;
let ultimoVideoTempo = -1, ultimoTs = 0;
let fps = 60, tAnt = performance.now();

function quadro(agora) {
  requestAnimationFrame(quadro);
  if (video.readyState < 2 || !video.videoWidth) return;

  const dt = Math.max((agora - tAnt) / 1000, 1e-3);
  tAnt = agora;
  fps = 0.92 * fps + 0.08 / dt;

  // tamanho do canvas em pixels reais (nítido em telas HD/retina)
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = Math.round(window.innerWidth * dpr), H = Math.round(window.innerHeight * dpr);
  if (tela.width !== W || tela.height !== H) { tela.width = W; tela.height = H; }

  // vídeo cobrindo a tela (object-fit: cover), espelhado
  const vw = video.videoWidth, vh = video.videoHeight;
  const s = Math.max(W / vw, H / vh);
  const cover = { dw: vw * s, dh: vh * s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2 };
  const k = REF_ALTURA / vh;                  // vídeo -> espaço de referência
  const paraTela = ([x, y]) => [W - (cover.ox + (x / k) * s), cover.oy + (y / k) * s];

  // detecção só quando chega um quadro novo da câmera
  if (video.currentTime !== ultimoVideoTempo) {
    ultimoVideoTempo = video.currentTime;
    const ts = Math.max(agora, ultimoTs + 1);
    ultimoTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    const det = res.landmarks.map((lms, i) => ({
      ref: lms.map((l) => [l.x * vw * k, l.y * vh * k]),
      // o detector recebe a imagem sem espelhar, então o lado vem invertido
      lado: res.handedness[i]?.[0]?.categoryName === "Right" ? "esquerda" : "direita",
    }));
    det.forEach((d) => { d.xTela = paraTela(d.ref[0])[0]; });
    det.sort((a, b) => a.xTela - b.xTela);
    if (det.length !== maos.length) maos = det.map(() => new Mao());
    det.forEach((d, i) => maos[i].atualizar(d.ref, d.lado, ts / 1000));
  }

  const ptsTela = maos.map((m) => m.ref.map(paraTela));
  let pontas = [];
  maos.forEach((m, i) => DEDOS.forEach(([, ponta], d) => { if (m.estados[d]) pontas.push(ptsTela[i][ponta]); }));
  if (!maos.length && DEMO) pontas = pontasDemo(DEMO, W, H, agora);
  const { chave, poly } = classificarForma(pontas);

  // só troca de filtro depois que a forma fica estável por alguns quadros
  if (!chave) { formaAtual = null; candidata = null; contagem = 0; }
  else if (chave !== formaAtual) {
    if (chave === candidata) contagem++;
    else { candidata = chave; contagem = 1; }
    if (contagem >= SHAPE_STABLE_FRAMES) { formaAtual = chave; candidata = null; contagem = 0; fadeInicio = agora; }
  } else { candidata = null; contagem = 0; }

  // vídeo de fundo
  ctx.setTransform(-1, 0, 0, 1, W, 0);
  ctx.drawImage(video, cover.ox, cover.oy, cover.dw, cover.dh);
  ctx.setTransform(1, 0, 0, 1, 0, 0);

  // filtro dentro da forma
  if (formaAtual && poly) {
    const f = FORMA[formaAtual];
    const t = Math.min(1, (agora - fadeInicio) / 1000 / FADE);
    const opac = t * t * (3 - 2 * t);
    const bb = caixa(poly);
    const centro = [poly.reduce((a, p) => a + p[0], 0) / poly.length, poly.reduce((a, p) => a + p[1], 0) / poly.length];
    const imagem = filtros.render(video, W, H, cover, f.modo, centro, dpr * (H / dpr > 900 ? 1.15 : 1), bb);

    ctx.save();
    caminho(poly);
    ctx.clip();
    ctx.globalAlpha = opac;
    ctx.drawImage(imagem, 0, 0);
    ctx.restore();

    // conector: brilho suave + linha fina
    ctx.lineJoin = "round";
    caminho(poly);
    ctx.strokeStyle = f.cor;
    ctx.globalAlpha = 0.16 * opac;
    ctx.lineWidth = 6 * dpr;
    ctx.stroke();
    ctx.globalAlpha = 0.95 * opac;
    ctx.lineWidth = dpr;
    ctx.stroke();
    ctx.globalAlpha = 1;

    ctx.globalAlpha = opac;
    texto(`${f.nome}  ·  ${f.filtro}`, bb.x0, bb.y0 - 30 * dpr, 19, f.cor, 600, dpr);
    ctx.globalAlpha = 1;
  }

  maos.forEach((m, i) => desenharMao(ptsTela[i], m, dpr));

  if (!maos.length) {
    ctx.textAlign = "center";
    const livre = ui.painel.classList.contains("oculto") || W / dpr <= 760 ? W : W - (420 + 28) * dpr;
    texto("Mostre as mãos para a câmera", livre / 2, H - 70 * dpr, 20, "rgba(255,255,255,0.9)", 300, dpr);
    ctx.textAlign = "left";
  }
  texto(`${Math.round(fps)} fps`, 20 * dpr, 30 * dpr, 12, "rgba(255,255,255,0.6)", 300, dpr);

  painelForma(formaAtual);
  painelRodape(pontas.length, maos);
}

// ---------------------------------------------------------------------------
// Início
// ---------------------------------------------------------------------------

async function criarDetector() {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const opcoes = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  try {
    return await HandLandmarker.createFromOptions(fileset, opcoes("GPU"));
  } catch {
    return await HandLandmarker.createFromOptions(fileset, opcoes("CPU"));
  }
}

async function iniciar() {
  ui.btnIniciar.disabled = true;
  ui.status.classList.remove("erro");
  try {
    if (!navigator.mediaDevices?.getUserMedia) throw new Error("Este navegador não permite acessar a câmera (use HTTPS).");
    ui.status.textContent = "Pedindo acesso à câmera…";
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1920 }, height: { ideal: 1080 }, frameRate: { ideal: 60 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();

    ui.status.textContent = "Carregando o modelo de mãos…";
    filtros = new Filtros();
    [landmarker] = await Promise.all([criarDetector(), document.fonts?.ready]);

    ui.inicio.classList.add("saindo");
    setTimeout(() => ui.inicio.remove(), 700);
    requestAnimationFrame(quadro);
  } catch (e) {
    console.error(e);
    const negado = e?.name === "NotAllowedError";
    ui.status.textContent = negado ? "Acesso à câmera negado. Libere a permissão no navegador e tente de novo." : `Erro: ${e.message || e}`;
    ui.status.classList.add("erro");
    ui.btnIniciar.disabled = false;
  }
}

ui.btnIniciar.addEventListener("click", iniciar);
if (DEMO) iniciar();
