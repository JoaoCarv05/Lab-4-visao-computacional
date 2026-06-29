"""
Gera imagens sintéticas de um tabuleiro de xadrez visto de diferentes ângulos.
Utilizado para demonstrar o pipeline de calibração quando não há webcam disponível.
"""
import cv2
import numpy as np
import os

def gerar_tabuleiro(rows, cols, square_size, img_w, img_h):
    """Cria uma imagem de tabuleiro de xadrez com as dimensões especificadas."""
    board = np.ones((rows * square_size, cols * square_size), dtype=np.uint8) * 255
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 1:
                y0 = r * square_size
                x0 = c * square_size
                board[y0:y0 + square_size, x0:x0 + square_size] = 0
    return board


def projetar_tabuleiro(board_img, K, rvec, tvec, board_corners_3d, img_size):
    """Projeta o tabuleiro usando os parâmetros da câmera e gera uma imagem sintética."""
    h_board, w_board = board_img.shape[:2]

    src_pts = np.array([
        [0, 0],
        [w_board, 0],
        [w_board, h_board],
        [0, h_board]
    ], dtype=np.float32)

    board_3d_corners = np.array([
        [0, 0, 0],
        [w_board, 0, 0],
        [w_board, h_board, 0],
        [0, h_board, 0]
    ], dtype=np.float32)

    dist_coeffs = np.zeros(5)
    img_pts, _ = cv2.projectPoints(board_3d_corners, rvec, tvec, K, dist_coeffs)
    dst_pts = img_pts.reshape(-1, 2).astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    img_out = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 200

    gray_bg = np.random.randint(180, 220, (img_size[1], img_size[0]), dtype=np.uint8)
    img_out[:, :, 0] = gray_bg
    img_out[:, :, 1] = gray_bg
    img_out[:, :, 2] = gray_bg

    board_color = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    warped = cv2.warpPerspective(board_color, M, img_size,
                                  borderMode=cv2.BORDER_TRANSPARENT)

    mask = cv2.warpPerspective(
        np.ones_like(board_img) * 255, M, img_size
    )

    mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    img_out = np.where(mask_3ch > 0, warped, img_out)

    noise = np.random.normal(0, 3, img_out.shape).astype(np.int16)
    img_out = np.clip(img_out.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img_out


def main():
    ROWS = 7
    COLS = 10
    SQUARE = 40

    IMG_W = 640
    IMG_H = 480

    fx = 600.0
    fy = 600.0
    cx = IMG_W / 2.0
    cy = IMG_H / 2.0
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)

    board_img = gerar_tabuleiro(ROWS, COLS, SQUARE, IMG_W, IMG_H)

    h_board, w_board = board_img.shape[:2]

    poses = [
        (np.array([0.0, 0.0, 0.0]),    np.array([-w_board/2, -h_board/2, 800.0])),
        (np.array([0.3, 0.0, 0.0]),    np.array([-w_board/2, -h_board/2, 850.0])),
        (np.array([-0.3, 0.0, 0.0]),   np.array([-w_board/2, -h_board/2, 850.0])),
        (np.array([0.0, 0.3, 0.0]),    np.array([-w_board/2, -h_board/2, 900.0])),
        (np.array([0.0, -0.3, 0.0]),   np.array([-w_board/2, -h_board/2, 900.0])),
        (np.array([0.2, 0.2, 0.0]),    np.array([-w_board/2 + 30, -h_board/2 + 20, 750.0])),
        (np.array([-0.2, 0.2, 0.0]),   np.array([-w_board/2 - 30, -h_board/2 + 20, 780.0])),
        (np.array([0.15, -0.15, 0.1]), np.array([-w_board/2, -h_board/2, 820.0])),
        (np.array([-0.1, 0.25, -0.1]), np.array([-w_board/2 + 50, -h_board/2 - 30, 860.0])),
        (np.array([0.25, 0.1, 0.05]),  np.array([-w_board/2 - 20, -h_board/2 + 40, 770.0])),
        (np.array([0.0, 0.0, 0.2]),    np.array([-w_board/2, -h_board/2, 900.0])),
        (np.array([0.35, 0.15, 0.0]),  np.array([-w_board/2, -h_board/2, 950.0])),
        (np.array([-0.15, -0.2, 0.1]), np.array([-w_board/2 + 40, -h_board/2 + 50, 830.0])),
    ]

    out_dir = "samples"
    os.makedirs(out_dir, exist_ok=True)

    board_3d = None

    for i, (rvec_vals, tvec_vals) in enumerate(poses):
        rvec = rvec_vals.reshape(3, 1)
        tvec = tvec_vals.reshape(3, 1)

        img = projetar_tabuleiro(board_img, K, rvec, tvec, board_3d,
                                 (IMG_W, IMG_H))

        fname = os.path.join(out_dir, f"frm{i}.jpg")
        cv2.imwrite(fname, img)
        print(f"Salvo: {fname}")

    print(f"\n{len(poses)} imagens geradas em '{out_dir}/'")
    print(f"Parâmetros da câmera simulada:")
    print(f"  fx = {fx}, fy = {fy}")
    print(f"  cx = {cx}, cy = {cy}")
    print(f"  Matriz K = \n{K}")


if __name__ == "__main__":
    main()
