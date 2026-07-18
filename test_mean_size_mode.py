import numpy as np

def logu(lo, hi, size, rng):
    return np.exp(rng.uniform(np.log(lo), np.log(hi), size=size))

def max_share(lo, hi, n_models, n_trials=2000):
    shares = []
    for t in range(n_trials):                          # lặp 2000 lần độc lập
        r = np.random.default_rng(1000 + t)             # rng riêng cho mỗi lần
        params = logu(lo, hi, n_models, r)              # rút n_models giá trị params_b (tỷ tham số)
        # breakpoint()
        ckpt = params * 2.0                             # đổi sang GB checkpoint (giống công thức thật)
        pool_total = ckpt.sum()                         # tổng chi phí nếu "mua" toàn bộ pool
        budget = 0.15 * pool_total                       # ngân sách = 15% tổng đó
        shares.append(ckpt.max() / budget)              # model đắt nhất chiếm bao nhiêu % ngân sách
    return np.array(shares)                             # mảng 2000 giá trị share

print(np.mean(max_share(0.5, 70.0, 60)))