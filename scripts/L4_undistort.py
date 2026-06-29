"""
L4_undistort.py — Correção de distorção usando cv2.undistort()

Utiliza os parâmetros de calibração previamente obtidos para corrigir
a distorção radial e tangencial de imagens capturadas.

Uso:
    python3 L4_undistort.py <arquivo_calibracao.npz> <imagem1.jpg> [imagem2.jpg ...]

Exemplo:
    python3 L4_undistort.py calibracao_samples.npz samples/frm0.jpg samples/frm5.jpg
"""
import cv2
import numpy as np
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 L4_undistort.py <calibracao.npz> <imagem1.jpg> [imagem2.jpg ...]")
        sys.exit(1)

    calib_file = sys.argv[1]
    image_files = sys.argv[2:]

    data = np.load(calib_file)
    mtx = data['mtx']
    dist = data['dist']

    print(f"Parâmetros carregados de: {calib_file}")
    print(f"K = \n{mtx}")
    print(f"dist = {dist}\n")

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro: não foi possível carregar {img_path}")
            continue

        h, w = img.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        # Método 1: cv2.undistort()
        dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

        # Recortar usando ROI
        x, y, w2, h2 = roi
        if w2 > 0 and h2 > 0:
            dst_cropped = dst[y:y + h2, x:x + w2]
        else:
            dst_cropped = dst

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_full = f"{base}_undistort.jpg"
        out_crop = f"{base}_undistort_crop.jpg"

        cv2.imwrite(out_full, dst)
        cv2.imwrite(out_crop, dst_cropped)

        print(f"{img_path}:")
        print(f"  Salvo (completo):  {out_full}")
        print(f"  Salvo (recortado): {out_crop}")


if __name__ == "__main__":
    main()
