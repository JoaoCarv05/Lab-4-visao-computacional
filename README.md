# Laboratório 4 – Calibração de Câmeras

**Visão Computacional 2026.2**

## Descrição

Este repositório contém o relatório e os códigos do Laboratório 4 da disciplina de Visão Computacional, focado em **calibração de câmeras** — estimação de parâmetros intrínsecos (matriz K), extrínsecos (R, t), coeficientes de distorção e correção de distorção de imagens utilizando OpenCV.

## Conteúdo

- `Lab4_Calibracao.ipynb` — Relatório completo em formato Jupyter Notebook
- `scripts/L4_cal.py` — Script de calibração de câmera (fornecido pelo professor)
- `scripts/L4_chess.py` — Script de captura de imagens do tabuleiro (alterado para salvar com prefixo)
- `scripts/L4_undistort.py` — Correção de distorção usando `cv2.undistort()`
- `scripts/L4_undistort_remap.py` — Correção de distorção usando remapping
- `gerar_imagens_sinteticas.py` — Gerador de imagens sintéticas do tabuleiro para demonstração
- `samples/` — Imagens de calibração (tabuleiro de xadrez)
- `imagens/` — Imagens geradas durante os experimentos
- `pattern.pdf` — Padrão de calibração (tabuleiro) para impressão

## Requisitos

```bash
pip install opencv-python opencv-contrib-python numpy matplotlib jupyter
```

## Execução

### Notebook (relatório completo)
```bash
jupyter notebook Lab4_Calibracao.ipynb
```

### Calibração com imagens de exemplo
```bash
cd samples
python3 ../scripts/L4_cal.py
```

### Captura de imagens para calibração (requer webcam)
```bash
python3 scripts/L4_chess.py
```

### Correção de distorção
```bash
# Método 1: cv2.undistort()
python3 scripts/L4_undistort.py calibracao_samples.npz samples/frm0.jpg

# Método 2: Remapping
python3 scripts/L4_undistort_remap.py calibracao_samples.npz samples/frm0.jpg
```
