# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 12:53:40 2026

@author: qhaid
"""
import numpy as np

def algorithm_1(A, b, priority_order=None):
    """
    时间序列字典序优化算法 (Algorithm 1)
    参数:
        A: list of 2D arrays, A[t][i,j] 为时间t的带宽矩阵，形状 (m, n)
        b: list of 1D arrays, b[t][i] 为时间t的需求向量，形状 (m,)
        priority_order: 变量优先级列表，例如 [0,1,2,...] 表示 x0->x1->x2...
                        默认按自然顺序 0,1,...,n-1
    返回:
        x_opt: list of 1D arrays, 每个时间t的最优解向量，形状 (n,)
        如果系统不一致，返回 None
    """
    T = len(A)                     # 时间点个数
    m, n = A[0].shape               # 用户数 m, 终端数 n
    if priority_order is None:
        priority_order = list(range(n))   # 默认优先级 x0 -> x1 -> ...

    # Step 1: 一致性检查
    for t in range(T):
        for i in range(m):
            if np.max(A[t][i, :]) < b[t][i] - 1e-12:   # 考虑浮点误差
                print(f"系统不一致：时间 {t}, 用户 {i} 无法满足需求")
                return None
    # 初始化解矩阵：每一行对应一个时间点，每一列对应一个变量
    x_opt = np.zeros((T, n))

    # Step 2: 按优先级依次确定变量
    for k_idx, k in enumerate(priority_order):
        # k 是当前要处理的变量的原始索引
        # 已处理的变量索引集合 higher = priority_order[:k_idx]
        higher = priority_order[:k_idx]
        for t in range(T):
            # 构建测试向量 y：前 k_idx 个分量用已确定的值，第 k 个分量为 0，其余为 1
            y = np.ones(n) * 1.0
            # 已确定的更高优先级变量
            for idx, var in enumerate(higher):
                y[var] = x_opt[t, var]
            # 当前变量置 0
            y[k] = 0.0

            # 计算哪些约束已被满足
            satisfied = []
            for i in range(m):
                # 计算 max_j (a_ij(t) ∧ y_j)
                val = np.max(np.minimum(A[t][i, :], y))
                if val >= b[t][i] - 1e-12:
                    satisfied.append(i)

            if len(satisfied) == m:
                # 所有约束都已满足，当前变量可取 0
                x_opt[t, k] = 0.0
            else:
                # 找出未满足的约束集
                unsatisfied = [i for i in range(m) if i not in satisfied]
                # 当前变量需要设为未满足约束中需求的最大值
                x_opt[t, k] = np.max([b[t][i] for i in unsatisfied])
    return x_opt

def main():
    # 定义例子1的数据
    # 时间点 T = {1,2,3}，用索引 0,1,2 表示
    A = []
    b = []

    # A(1) 矩阵 (6x6)
    A1 = np.array([
        [0.4, 0.5, 0.6, 0.6, 0.3, 0.55],
        [0.6, 0.7, 0.5, 0.55, 0.45, 0.5],
        [0.5, 0.45, 0.6, 0.55, 0.4, 0.3],
        [0.7, 0.5, 0.6, 0.4, 0.8, 0.6],
        [0.6, 0.65, 0.45, 0.5, 0.7, 0.55],
        [0.6, 0.4, 0.3, 0.7, 0.4, 0.45]
    ])
    A.append(A1)

    # A(2) 矩阵
    A2 = np.array([
        [0.5, 0.6, 0.7, 0.5, 0.4, 0.6],
        [0.7, 0.8, 0.6, 0.6, 0.5, 0.55],
        [0.6, 0.5, 0.7, 0.6, 0.5, 0.4],
        [0.8, 0.6, 0.7, 0.5, 0.9, 0.7],
        [0.7, 0.7, 0.5, 0.6, 0.8, 0.6],
        [0.7, 0.5, 0.4, 0.8, 0.5, 0.5]
    ])
    A.append(A2)

    # A(3) 矩阵
    A3 = np.array([
        [0.45, 0.55, 0.65, 0.55, 0.35, 0.5],
        [0.65, 0.75, 0.55, 0.5, 0.5, 0.45],
        [0.55, 0.5, 0.65, 0.5, 0.45, 0.35],
        [0.75, 0.55, 0.65, 0.45, 0.85, 0.65],
        [0.65, 0.7, 0.5, 0.55, 0.75, 0.6],
        [0.65, 0.45, 0.35, 0.75, 0.45, 0.5]
    ])
    A.append(A3)

    # 需求向量 b(1), b(2), b(3)
    b1 = np.array([0.5, 0.6, 0.55, 0.7, 0.65, 0.5])
    b2 = np.array([0.55, 0.65, 0.6, 0.75, 0.7, 0.55])
    b3 = np.array([0.52, 0.62, 0.57, 0.72, 0.67, 0.52])
    b = [b1, b2, b3]

    # 默认优先级 x0->x1->...->x5 (对应 x1->x2->...->x6)
    x_opt = algorithm_1(A, b)

    if x_opt is not None:
        print("算法1求解得到的最优解 x*(t):")
        for t, vec in enumerate(x_opt):
            print(f"t = {t+1}: {np.round(vec, 4)}")
    else:
        print("系统不一致，无解。")

    # 验证论文中给出的解
    print("\n论文中给出的解：")
    print("t=1: [0.0, 0.6, 0.0, 0.55, 0.7, 0.0]")
    print("t=2: [0.0, 0.65, 0.0, 0.6, 0.75, 0.0]")
    print("t=3: [0.0, 0.62, 0.57, 0.52, 0.72, 0.0]")

if __name__ == "__main__":
    main()
