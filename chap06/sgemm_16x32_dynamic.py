import math
import time
from time import clock_gettime, CLOCK_MONOTONIC
from argparse import ArgumentParser
import numpy as np
from numpy.typing import NDArray

from videocore6.assembler import qpu
from videocore6.driver import Driver


def getsec() -> float:
    return clock_gettime(CLOCK_MONOTONIC)


def get_thid() -> None:
    # QPUのIDとスレッドのIDを取り出してr0に格納する．
    # 8つのスレッドが並列で動作しており，スレッドによって異なる結果が返る．
    tidx(r0)

    # QPUのIDを抜き出す
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)


@qpu
def kernel(asm, num_qpus: int) -> None:
    A_ADDR = 0  # Aの先頭アドレス
    A_STR = 1  # q * 4
    B_ADDR = 2  # Bの先頭アドレス
    B_STR = 3  # r * 4
    C_ADDR = 4  # Cの先頭アドレス
    HBLOCK = 5  # 1スレッドが処理するブロックの高さ
    WBLOCK = 6  # 1スレッドが処理するブロックの幅
    LOOP_I = 7  # 列方向のループ回数
    LOOP_J = 8  # 行方向のループ回数
    LOOP_K = 9  # 畳み込みのループ回数
    FRAC_H = 10  # 行方向の端数
    FRAC_W = 11  # 列方向の端数

    eidx(r0).mov(r2, 0)  # r0 = [0, 1, ..., 15], r2 = [0] * 16
    for idx in [
        A_ADDR,
        A_STR,
        B_ADDR,
        B_STR,
        C_ADDR,
        HBLOCK,
        WBLOCK,
        LOOP_I,
        LOOP_J,
        LOOP_K,
        FRAC_H,
        FRAC_W,
    ]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond="pushz")
        mov(r2, r5, cond="ifa")
        # r2 = [A_ADDR, A_STR, B_ADDR, B_STR, C_ADDR, HBLOCK, WBLOCK, LOOP_I, LOOP_J, LOOP_K, FRAC_H, FRAC_W, 0, 0, 0, 0]

    if num_qpus == 1:
        mov(r0, 0)  # r0 = [0] * 16
    elif num_qpus == 8:
        get_thid()  # r0 = [thid] * 16
    else:
        raise Exception("num_qpus must be 1 or 8")

    # 全ての汎用レジスタをゼロ初期化
    for i in range(64):
        mov(rf[i], 0.0)

    eidx(r4)  # r4 = [0, 1, ..., 15]

    # B set
    # 横に4分割したうちの1つを使う
    sub(r1, r0, 4, cond="pushn")  # r1 = [thid - 4] * 16
    mov(r1, r0, cond="ifa")  # r1 = [thid] * 16 or [thid - 4] * 16: 担当するブロックの列番号
    rotate(broadcast, r2, -WBLOCK)  # r5 = [WBLOCK] * 16
    umul24(r3, r5, r1)  # r3 = [thid * WBLOCK] * 16 or [(thid - 4) * WBLOCK] * 16: 担当するブロックの開始位置のX座標

    # 端数処理
    # スレッドIDが3または7，つまり右端のブロックを担当しているならば，はみ出た分だけブロックの開始位置を左にずらす
    rotate(broadcast, r2, -FRAC_W)  # r5 = [FRAC_W] * 16
    sub(null, r1, 3, cond="pushz")
    sub(r3, r3, r5, cond="ifa")

    mov(r1, r3)  # r1 = [thid * WBLOCK] * 16 or [(thid - 4) * WBLOCK] * 16: 担当するブロックの開始位置のX座標

    # Bの担当ブロックの開始位置に移動
    sub(null, r4, B_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")

    # A set
    # 縦に2分割したうちの1つを使う
    shr(r3, r0, 2)  # r3: QPU0~3 -> 0, QPU4~7 -> 1
    rotate(broadcast, r2, -HBLOCK)  # r5 = [HBLOCK] * 16
    mov(r0, r5)  # r0 = [HBLOCK] * 16
    rotate(broadcast, r2, -FRAC_H)  # r5 = [FRAC_H] * 16
    sub(r0, r0, r5)  # r0 = [HBLOCK - FRAC_H] * 16

    umul24(r3, r0, r3)  # r3: QPU0~3 -> 0, QPU4~7 -> HBLOCK - FRAC_H: 担当するブロックの開始位置のY座標
    rotate(broadcast, r2, -B_STR)  # r5 = [B_STR] * 16
    umul24(r0, r3, r5)  # r0: Cの担当するブロックの開始位置と同じ行の左端までのオフセット

    # Aの担当ブロックの開始位置に移動
    rotate(broadcast, r2, -A_STR)  # r5 = [A_STR] * 16
    umul24(r3, r5, r3)  # r3: Aの開始位置までのオフセット
    sub(null, r4, A_ADDR, cond="pushz")
    add(r2, r2, r3, cond="ifa")

    # Cの担当ブロックの開始位置に移動
    add(r1, r1, r0)  # r1: Cの担当するブロックの開始位置までのオフセット
    sub(null, r4, C_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")

    # 使用する変数，定数のエイリアス
    iidx = rf50
    jidx = rf51
    kidx = rf52
    istp = rf53
    jstp = rf54
    a_cur = rf55
    b_cur = rf56
    c_cur = rf57
    simd_stp = rf58
    ldi128 = rf59
    ldi16 = rf60
    mov(ldi128, 1)
    shl(ldi128, ldi128, 7)  # ldi128 = 128
    mov(ldi16, 1)
    shl(ldi16, ldi16, 4)  # ldi16 = 16
    mov(simd_stp, 1)
    shl(simd_stp, simd_stp, 6)  # simd_stp = 64 = 16 * 4
    mov(iidx, 0)  # iidx = 0

    with loop as iloop:
        # set a_cur
        # 16 x iidx x A_STR x eidx + A_ADDR
        umul24(r0, ldi16, iidx)  # r0 = iidx * 16

        # 端数処理
        # if HBLOCK - i * 16 < 0:
        #     i - (16 + (HBLOCK - i * 16))
        add(r1, r0, ldi16)  # r1 = (iidx + 1) * 16: AのY座標を進める
        rotate(broadcast, r2, -HBLOCK)  # r5 = HBLOCK
        sub(r1, r5, r1, cond="pushn")  # r1 = HBLOCK - (iidx + 1) * 16
        b(R.fraction_i_end, cond="anyna")  # r1 < 0 <=> HBLOCK < (iidx + 1) * 16 が成り立たないなら3命令後にジャンプ

        # 以下3行はどちらにせよ実行される
        # nopがもったいないので競合しない命令を差し込んでいる
        mov(r1, 0)  # r1 = 0 (nop)
        eidx(a_cur)  # a_cur = [0, 1, ..., 15] (nop)
        rotate(broadcast, r2, -A_STR)  # r5 = A_STR (nop)

        # if (iidx + 1) * 16 > HBLOCK: 現在のループで処理するAのブロックの終了位置のY座標が担当ブロックをはみ出る場合
        add(r0, r0, r1)  # r0 = HBLOCK - 16
        eidx(a_cur)  # TODO: 上と同じだから不要では？
        rotate(broadcast, r2, -A_STR)  # TODO: 上と同じだから不要では？
        # endif (iidx + 1) * 16 >= HBLOCK

        L.fraction_i_end

        umul24(a_cur, a_cur, r5)
        umul24(r0, r5, r0)
        add(a_cur, a_cur, r0)
        rotate(broadcast, r2, -A_ADDR)
        add(a_cur, a_cur, r5)
        mov(jidx, 0)
        with loop as jloop:
            # set b_cur
            # 1 : 32 x 4(float) x jidx
            umul24(r0, ldi128, jidx)

            # if WBLOCK - j * 32 * 4(bytes) < 0:
            #     r0 - (128 + (WBLOCK - j * 128)
            add(r3, r0, ldi128)
            rotate(broadcast, r2, -WBLOCK)
            sub(r3, r5, r3, cond="pushn")
            b(R.fraction_j_end, cond="anyna")
            mov(kidx, 0)
            eidx(b_cur)
            mov(rf49, 0)

            add(r0, r0, r3)
            mov(rf49, r3)
            mov(kidx, 0)
            eidx(b_cur)

            L.fraction_j_end

            # 2 : eidx x 4 + B_ADDR
            shl(b_cur, b_cur, 2)
            rotate(broadcast, r2, -B_ADDR)
            add(b_cur, b_cur, r5)

            # 1 + 2
            add(b_cur, b_cur, r0)

            with loop as kloop:
                mov(tmua, a_cur, sig=thrsw)
                add(a_cur, a_cur, 4)  # nop()
                add(kidx, kidx, 1)  # nop()
                nop(sig=ldtmu(r4))
                for lj in range(2):
                    stp = lj * 16
                    mov(tmua, b_cur, sig=thrsw)
                    if lj == 0:
                        add(b_cur, b_cur, simd_stp)  # nop()
                    else:
                        nop()
                    nop()
                    nop(sig=ldtmu(r3))
                    rotate(broadcast, r4, 0)
                    fmul(r0, r5, r3)
                    for li in range(15):
                        rotate(broadcast, r4, -(li + 1))
                        fadd(rf[stp + li], rf[stp + li], r0).fmul(r0, r5, r3)
                    fadd(rf[stp + 15], rf[stp + 15], r0)
                rotate(broadcast, r2, -LOOP_K)
                sub(null, r5, kidx, cond="pushz")
                kloop.b(cond="anyna")
                sub(b_cur, b_cur, simd_stp)  # nop()
                rotate(broadcast, r2, -B_STR)  # nop()
                add(b_cur, b_cur, r5)  # nop()

            umul24(r0, ldi16, iidx)
            rotate(broadcast, r2, -B_STR)
            umul24(r0, r5, r0)

            eidx(c_cur)
            umul24(c_cur, c_cur, 4)

            umul24(r1, r1, r5)  # 端数処理

            add(c_cur, c_cur, r0)

            # 32 x 4(float) x jidx
            umul24(r0, ldi128, jidx)
            add(r0, r0, rf49)
            rotate(broadcast, r2, -C_ADDR)
            add(c_cur, c_cur, r5)
            add(c_cur, c_cur, r0)
            add(c_cur, c_cur, r1)  # 端数処理

            rotate(broadcast, r2, -B_STR)
            sub(r0, r5, simd_stp)
            for li in range(16):
                mov(tmud, rf[li])
                mov(tmua, c_cur)
                add(c_cur, c_cur, simd_stp)
                mov(rf[li], 0.0)
                tmuwt()
                mov(tmud, rf[li + 16])
                mov(tmua, c_cur)
                add(c_cur, c_cur, r0)
                mov(rf[li + 16], 0.0)
                tmuwt()

            rotate(broadcast, r2, -LOOP_J)
            add(jidx, jidx, 1)
            sub(null, r5, jidx, cond="pushz")
            jloop.b(cond="anyna")
            rotate(broadcast, r2, -A_STR)  # nop()
            sub(a_cur, a_cur, r5)  # nop()
            nop()

        add(iidx, iidx, 1)
        rotate(broadcast, r2, -LOOP_I)
        sub(null, r5, iidx, cond="pushz")
        iloop.b(cond="anyna")
        nop()
        nop()
        nop()

    barrierid(syncb, sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


def dot(A: NDArray, B: NDArray) -> NDArray:
    SEG_MAT_ROWS = 16
    SEG_MAT_COLS = 32

    assert A.shape[1] == B.shape[0]

    p = A.shape[0]
    q = A.shape[1]
    r = B.shape[1]

    assert p >= SEG_MAT_ROWS and r >= SEG_MAT_COLS

    if p < 32 or r < 128:
        num_thx = 1
        num_thy = 1
    else:
        num_thx = 4
        num_thy = 2

    num_qpus = num_thx * num_thy

    # 1スレッドが処理する範囲
    hblock = math.ceil(p / num_thy)
    wblock = math.ceil(r / num_thx)

    # スレッド分割の端数
    frac_h = hblock * num_thy - p
    frac_w = (wblock * num_thx - r) * 4

    # ループ回数
    j_idx = math.ceil(r / (SEG_MAT_COLS * num_thx))
    i_idx = math.ceil(p / (SEG_MAT_ROWS * num_thy))
    k_idx = q

    wblock *= 4  # float32 = 4bytes

    with Driver(data_area_size=1024 * 1024 * 1024 + 1024 * 1024 * 512) as drv:
        # メモリ確保
        A_ = drv.alloc((p, q), dtype="float32")
        B_ = drv.alloc((q, r), dtype="float32")
        C_ = drv.alloc((p, r), dtype="float32")
        A_[:] = A
        B_[:] = B

        # uniform setting
        unif = drv.alloc(16, dtype="uint32")
        unif[0] = A_.addresses()[0, 0]  # Aの先頭アドレス
        unif[1] = A_.strides[0]  # q * 4
        unif[2] = B_.addresses()[0, 0]  # Bの先頭アドレス
        unif[3] = B_.strides[0]  # r * 4
        unif[4] = C_.addresses()[0, 0]  # Cの先頭アドレス
        unif[5] = hblock
        unif[6] = wblock
        unif[7] = i_idx
        unif[8] = j_idx
        unif[9] = k_idx
        unif[11] = frac_h
        unif[10] = frac_w
        code = drv.program(kernel, num_qpus=num_qpus)
        drv.execute(code, unif.addresses()[0], thread=num_qpus)

        return np.array(C_)


def main():
    parser = ArgumentParser()
    parser.add_argument("--p", help="p size", type=int, default=1024)
    parser.add_argument("--q", help="q size", type=int, default=1024)
    parser.add_argument("--r", help="r size", type=int, default=1024)
    parser.add_argument("--iter", help="The number of iterations", type=int, default=1)
    args = parser.parse_args()
    p = args.p
    q = args.q
    r = args.r
    iter = args.iter

    A = np.random.rand(p, q) * 0.1
    B = np.random.rand(q, r) * 0.1

    # Run the program
    cpu_time_sum = 0.0  # [s]
    for i in range(iter):
        print(f"CPU Iteration {i + 1}")
        t_start = time.time()
        C_ref = np.dot(A, B)
        t_end = time.time()
        cpu_time_sum += t_end - t_start
    cpu_time_avg = cpu_time_sum / iter

    gpu_time_sum = 0.0  # [s]
    for i in range(iter):
        print(f"GPU Iteration {i + 1}")
        t_start = time.time()
        C = dot(A, B)
        t_end = time.time()
        gpu_time_sum += t_end - t_start
    gpu_time_avg = gpu_time_sum / iter

    def gflops(time: float) -> float:
        return p * q * r * 2 / time * 1e-9

    print(np.count_nonzero(C != 1))
    print(f"CPU time:  {cpu_time_avg * 1000:.2f} ms")
    print(f"GPU time:  {gpu_time_avg * 1000:.2f} ms")
    print(f"CPU FLOPS: {gflops(cpu_time_avg):.2f} GFLOPS")
    print(f"GPU FLOPS: {gflops(gpu_time_avg):.2f} GFLOPS")
    print("minimum absolute error: {:.4e}".format(float(np.min(np.abs(C_ref - C)))))
    print("maximum absolute error: {:.4e}".format(float(np.max(np.abs(C_ref - C)))))
    print("minimum relative error: {:.4e}".format(float(np.min(np.abs((C_ref - C) / C_ref)))))
    print("maximum relative error: {:.4e}".format(float(np.max(np.abs((C_ref - C) / C_ref)))))


if __name__ == "__main__":
    main()
