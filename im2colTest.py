import numpy as np


def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    input_data : (データ数, チャンネル, 高さ, 幅)の4次元配列からなる入力データ
    filter_h : フィルターの高さ
    filter_w : フィルターの幅
    stride : ストライド
    pad : パディング

    Returns
    -------
    col : 2次元配列
    """
    N, C, H, W = input_data.shape
    print("\ninput_data.shape: " + str(input_data.shape))

    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1
    print("\nout_h: " + str(out_h))
    print("out_w: " + str(out_w))

    img = np.pad(input_data, [(0, 0), (0, 0),
                              (pad, pad), (pad, pad)], 'constant')
    print("\n\nimg: " + str(img))

    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    # print("\n\ncol: " + str(col))

    for y in range(filter_h):
        y_max = y + stride * out_h
        print("\ny_max: " + str(y_max))

        for x in range(filter_w):
            x_max = x + stride * out_w
            print("x_max: " + str(x_max))

            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
            # print("\n\nimg[:, :, y:y_max:stride, x:x_max:stride]: " +
            #     str(img[:, :, y:y_max:stride, x:x_max:stride]))
            print("\ncol: " + str(col[:, :, y, x, :, :]))

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    # print("\n\ncol.transpose: " + str(col))

    return col


inputData = np.random.randn(1, 5, 4, 4)

im2col(inputData, 2, 2)
