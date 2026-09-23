# Filtro com as mãos ✋🎨

Aplicação em Python que usa a webcam para **identificar as mãos e cada um dos dedos** e, ligando as pontas dos dedos levantados, **desenha uma forma** no vídeo. A imagem dentro da forma recebe um **filtro diferente para cada forma**: pontilhado, cartoon, térmico, neon e outros.

Tudo roda em tempo real numa interface Full HD, com um painel lateral que mostra a legenda de cada forma e os dedos detectados.

<!-- Coloque aqui um print da aplicação: ![Demonstração](docs/demo.png) -->

## Funcionalidades

- Detecta até **2 mãos** e identifica **polegar, indicador, médio, anelar e mínimo**, além de dizer se é a mão esquerda ou a direita.
- Liga as pontas dos dedos levantados e forma **círculo, triângulo, quadrado, retângulos e polígonos de 5 a 10 lados**.
- Aplica **11 filtros**, um por forma, só dentro da área desenhada, com borda suave e transição em fade.
- Traz um **painel lateral** em vidro fosco com a legenda das formas, a forma ativa em destaque e os dedos levantados de cada mão.
- Mantém o **vídeo fluido**: a câmera é lida numa thread separada, a detecção roda em paralelo e os pontos são suavizados com o filtro *One Euro*.

## Formas e filtros

| Dedos levantados | Forma | Filtro |
| :---: | --- | --- |
| 02 | Círculo (polegar + indicador) | Lupa (zoom 2x) |
| 03 | Triângulo | Pontilhado preto e branco (retícula de jornal) |
| 04 | Quadrado | Pontilhado colorido (pop-art) |
| 04 | Retângulo deitado | Cartoon |
| 04 | Retângulo em pé | Azul (cianotipia) |
| 05 | Pentágono (uma mão aberta) | Esboço a lápis |
| 06 | Hexágono (3 dedos em cada mão) | Térmico |
| 07 | Heptágono | Pixelado |
| 08 | Octógono (4 dedos em cada mão) | Negativo |
| 09 | Eneágono | Sépia |
| 10 | Decágono (duas mãos abertas) | Neon |

Com 4 dedos, a forma depende da proporção da moldura: quase igual nos dois lados vira **quadrado**, mais larga vira **retângulo deitado** e mais alta vira **retângulo em pé**.

> Dica: para fazer uma moldura, use o polegar e o indicador das duas mãos, como quem enquadra uma foto.

## Requisitos

- Python **3.10+** (testado no 3.12)
- Uma webcam
- Windows, Linux ou macOS (a interface foi pensada para o Windows, com a fonte Segoe UI)

## Instalação

```bash
git clone https://github.com/icrcode/indentificador-de-items.git
cd indentificador-de-items

# (opcional) ambiente virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/macOS

pip install -r requirements.txt
```

## Como usar

```bash
python main.py
```

Na primeira execução, o programa baixa sozinho o modelo de detecção de mãos do MediaPipe (`hand_landmarker.task`, cerca de 8 MB).

### Teclas

| Tecla | Ação |
| --- | --- |
| `F` | Liga ou desliga a tela cheia |
| `H` | Mostra ou esconde o painel lateral |
| `Q` / `ESC` | Sai |

## Ajustes

As configurações ficam no topo do [main.py](main.py):

| Constante | O que faz |
| --- | --- |
| `LARGURA`, `ALTURA` | Resolução da interface (padrão 1920×1080) |
| `MIN_CUTOFF`, `BETA` | Suavização dos pontos. Diminua `MIN_CUTOFF` se o contorno tremer e aumente `BETA` se ele demorar a acompanhar a mão |
| `DEDO_SOBE`, `DEDO_DESCE` | Quão esticado o dedo precisa estar para contar como levantado |
| `POLEGAR_SOBE`, `POLEGAR_DESCE` | O mesmo, só para o polegar |
| `SQUARE_RATIO` | Até que proporção 4 pontos ainda contam como quadrado |
| `SHAPE_STABLE_FRAMES` | Quantos quadros a forma precisa se manter para o filtro trocar |
| `FADE` | Duração da transição entre filtros, em segundos |

Para trocar o filtro de uma forma, ou criar um filtro novo, edite a lista `FORMAS`. Um filtro é só uma função que recebe uma imagem BGR e devolve outra do mesmo tamanho:

```python
def filtro_meu(img):
    return cv2.GaussianBlur(img, (0, 0), 8)
```

## Como funciona

1. **Captura**: a webcam é lida numa thread própria, que guarda sempre o quadro mais recente.
2. **Detecção**: o [MediaPipe Hand Landmarker](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker) encontra 21 pontos em cada mão, em modo assíncrono (`LIVE_STREAM`).
3. **Dedos**: um dedo conta como levantado quando a ponta está bem mais longe do pulso do que a articulação do meio. Para o polegar, a referência é a distância até a base do indicador.
4. **Forma**: as pontas dos dedos levantados são ordenadas pelo ângulo em volta do centro e formam um polígono. A quantidade de pontas define a forma.
5. **Filtro**: o filtro é aplicado só no retângulo que envolve a forma e depois recortado por uma máscara com borda suave.
6. **Interface**: textos e painel são desenhados com o Pillow em alta resolução, reduzidos e guardados em cache. A cada quadro, só são colados sobre o vídeo.

## Estrutura

```text
.
├── main.py             # aplicação completa
├── requirements.txt    # dependências
├── hand_landmarker.task  # modelo do MediaPipe (baixado automaticamente, fora do git)
└── LICENSE
```

## Tecnologias

- [MediaPipe](https://developers.google.com/mediapipe) — detecção das mãos
- [OpenCV](https://opencv.org/) — câmera, filtros e desenho
- [NumPy](https://numpy.org/) — cálculos com as imagens
- [Pillow](https://python-pillow.org/) — textos com acentos e painel

## Licença

Distribuído sob a licença MIT. Veja [LICENSE](LICENSE).

Feito por Ícaro Caldeira Botelho.
