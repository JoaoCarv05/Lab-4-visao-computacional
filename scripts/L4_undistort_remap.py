"""
L4_undistort_remap.py — Correção de distorção usando Remapping

Utiliza cv2.initUndistortRectifyMap() + cv2.remap() para corrigir
a distorção. Este método é mais eficiente para processar múltiplas
imagens, pois os mapas são calculados uma única vez.

Uso:
    python3 L4_undistort_remap.py <arquivo_calibracao.npz> <imagem1.jpg> [imagem2.jpg ...]

Exemplo:
    python3 L4_undistort_remap.py calibracao_samples.npz samples/frm0.jpg samples/frm5.jpg
"""
import cv2
import numpy as np
import sys
import os


def main():
    if len(sys.argv) < 3:
        print("Uso: python3 L4_undistort_remap.py <calibracao.npz> <imagem1.jpg> [imagem2.jpg ...]")
        sys.exit(1)

    calib_file = sys.argv[1]
    image_files = sys.argv[2:]

    data = np.load(calib_file)
    mtx = data['mtx']
    dist = data['dist']

    print(f"Parâmetros carregados de: {calib_file}")
    print(f"K = \n{mtx}")
    print(f"dist = {dist}\n")

    # Os mapas serão calculados na primeira imagem e reutilizados
    mapx = None
    mapy = None
    prev_size = None

    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erro: não foi possível carregar {img_path}")
            continue

        h, w = img.shape[:2]
        cur_size = (w, h)

        # Recalcular mapas apenas se a resolução mudar
        if cur_size != prev_size:
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
            mapx, mapy = cv2.initUndistortRectifyMap(
                mtx, dist, None, newcameramtx, (w, h), 5
            )
            prev_size = cur_size
            print(f"  Mapas de remapping recalculados para resolução {w}x{h}")

        # Método 2: Remapping
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Recortar usando ROI
        x, y, w2, h2 = roi
        if w2 > 0 and h2 > 0:
            dst_cropped = dst[y:y + h2, x:x + w2]
        else:
            dst_cropped = dst

        base = os.path.splitext(os.path.basename(img_path))[0]
        out_full = f"{base}_remap.jpg"
        out_crop = f"{base}_remap_crop.jpg"

        cv2.imwrite(out_full, dst)
        cv2.imwrite(out_crop, dst_cropped)

        print(f"{img_path}:")
        print(f"  Salvo (completo):  {out_full}")
        print(f"  Salvo (recortado): {out_crop}")


if __name__ == "__main__":
    main()
