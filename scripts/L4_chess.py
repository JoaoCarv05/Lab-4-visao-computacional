"""
L4_chess.py — Captura de imagens do tabuleiro de xadrez para calibração.

Uso:
    python3 L4_chess.py

Controles:
    's' — salvar o frame atual
    'q' — encerrar

Alterado para salvar na pasta webcam_joao/ com prefixo 'joao_'.
"""
import numpy as np
import cv2 as cv
import os

# Pasta de saída (altere o nome conforme o integrante da equipe)
OUTPUT_DIR = "webcam_joao"
PREFIX = "joao_"

os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

i = 0

print(f"Salvando imagens em: {OUTPUT_DIR}/")
print("Pressione 's' para salvar, 'q' para sair.")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Display the resulting frame
    cv.imshow('frame', frame)

    # Save on "s" key or exit on "q"
    k = cv.waitKey(1)
    if k == ord('s'):
        fname = os.path.join(OUTPUT_DIR, f"{PREFIX}{i}.jpg")
        cv.imwrite(fname, frame)
        i = i + 1
        print(f"Salvo: {fname} (frame {i})")
    elif k == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
print(f"\nTotal de imagens capturadas: {i}")
