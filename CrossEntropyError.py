import numpy as np

def cross_entropy_error(y:np.ndarray,t:np.ndarray):
    """
    交差エントロピー誤差

    Parameters
    ----------
    y : np.ndarray
        ニューラルネットワークの出力結果
    t : np.ndarray
        MNISTテストデータのラベル(one-hot表現)

    Returns
    -------
    -np.sum(t*np.log(y+delta)) : np.ndarray
        引数を交差エントロピー誤差で計算した結果

    """
    print("y.shape[0]:before "+str(y.shape[0]))
    print("y.shape:before "+str(y.shape))
    print("y.ndim:before "+str(y.ndim))
    if y.ndim==1:
        t=t.reshape(1,t.size)
        y=y.reshape(1,y.size)
        print("t.reshape(1,t.size):"+str(t))
        print("y.reshape(1,y.size):"+str(y))

    batch_size=y.shape[0]
    print("y.shape[0]:after "+str(y.shape[0]))
    print("y.shape:after "+str(y.shape))    
    print("y.ndim:after "+str(y.ndim))
    delta=1e-7
    return -np.sum(t*np.log(y+delta))/batch_size


t=np.array([0,0,1,0,0,0,0,0,0,0])
y=np.array([0.1,0.05,0.0,0.6,0.05,0.1,0.0,0.1,0.0,0.0])

print(cross_entropy_error(y,t))