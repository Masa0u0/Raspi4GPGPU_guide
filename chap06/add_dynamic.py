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


def read_add_write() -> None:
    # Aの16要素をrf5に読み出す
    # nop()2回分の時間を有効活用するために競合しない命令を差し込んでいる
    mov(tmua, r1, sig=thrsw)
    mov(r0, 1)  # r0 = [1] * 16 (nop)
    shl(r0, r0, 6)  # r0 = [64] * 16: 1つのQPUが1ループで計算するブロックのメモリ幅 = 16 * 4[byte] = 64[byte] (nop)
    nop(sig=ldtmu(rf5))

    # Bの16要素をrf6に読み出す
    # nop()2回分の時間を有効活用するために競合しない命令を差し込んでいる
    mov(tmua, r3, sig=thrsw)
    add(r1, r1, r0)  # Aのアドレスを1ループ分進める (nop)
    add(r3, r3, r0)  # Bのアドレスを1ループ分進める (nop)
    nop(sig=ldtmu(rf6))

    # A + Bの要素を計算してrf0に格納
    fadd(rf0, rf5, rf6)

    # A + Bの結果をCのアドレスに書き出す
    mov(tmud, rf0)
    mov(tmua, rf1)
    add(rf1, rf1, r0)  # Cのアドレスを1ループ分進める
    tmuwt()


def exit_qpu() -> None:
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


@qpu
def kernel(asm, num_qpus: int) -> None:
    A_ADDR = 0  # Aの先頭アドレス
    B_ADDR = 1  # Bの先頭アドレス
    C_ADDR = 2  # Cの先頭アドレス
    PROC_SIZE = 3  # QPU0~6それぞれが計算する要素数
    LOOP_NUM = 4  # QPU0~6のループ回数 (最後のループを除く)
    EDGE_MOD = 5  # QPU0~6の最後のループの16要素のうち要素なし部分の個数
    LOOP_NUM_LTH = 6  # QPU7のループ回数 (最後のループを除く)
    EDGE_MOD_LTH = 7  # QPU7の最後のループの16要素のうち要素なし部分の個数
    eidx(r0).mov(r2, 0)  # r0 = [0, 1, ..., 15], r2 = [0] * 16
    for idx in [
        A_ADDR,
        B_ADDR,
        C_ADDR,
        PROC_SIZE,
        LOOP_NUM,
        EDGE_MOD,
        LOOP_NUM_LTH,
        EDGE_MOD_LTH,
    ]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond="pushz")
        mov(r2, r5, cond="ifa")
        # r2 = [A_ADDR, B_ADDR, C_ADDR, PROC_SIZE, LOOP_NUM, EDGE_MOD, LOOP_NUM_LTH, EDGE_MOD_LTH, 0, 0, 0, 0, 0, 0, 0, 0]

    if num_qpus == 1:
        mov(r0, 0)  # r0 = [0] * 16
    elif num_qpus == 8:
        get_thid()  # r0 = [thid] * 16
    else:
        raise Exception("num_qpus must be 1 or 8")

    # 全ての汎用レジスタをゼロ初期化
    for i in range(64):
        mov(rf[i], 0.0)

    # スレッドIDが7でなければジャンプ
    sub(null, r0, 7, cond="pushz")
    b(R.not_th_id7, cond="anyna")

    # ===== スレッドIDが7の場合 =====
    nop()
    nop()
    nop()
    eidx(r3)  # r3 = [0, 1, ..., 15]

    rotate(broadcast, r2, -LOOP_NUM_LTH)  # r5 = [LOOP_NUM_LTH] * 16
    sub(null, r3, LOOP_NUM, cond="pushz")
    mov(r2, r5, cond="ifa")  # r2のLOOP_NUMをLOOP_NUM_LTHで置き換える

    rotate(broadcast, r2, -EDGE_MOD_LTH)  # r5 = [EDGE_MOD_LTH] * 16
    sub(null, r3, EDGE_MOD, cond="pushz")
    mov(r2, r5, cond="ifa")  # r2のEDGE_MODをEDGE_MOD_LTHで置き換える
    # ===== スレッドIDが7の場合 =====

    # 以下QPU0~6とQPU7を同じように扱える
    L.not_th_id7

    # 各行列に対する自スレッドのオフセットを計算
    rotate(broadcast, r2, -PROC_SIZE)  # r5 = [PROC_SIZE] * 16
    umul24(r1, r5, r0)  # r1 = [PROC_SIZE * thid] * 16
    shl(r1, r1, 2)  # r1 = [PROC_SIZE * thid * 4] * 16: Float32は4バイトだから1要素に対して4をかける
    nop()
    nop()

    # A_ADDR, B_ADDR, C_ADDRに，それぞれ自スレッドが担当するブロックの開始アドレスを代入
    eidx(r4)  # r4 = [0, 1, ..., 15]
    sub(null, r4, A_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")
    sub(null, r4, B_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")
    sub(null, r4, C_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")

    eidx(r0)  # r0 = [0, 1, ..., 15]
    shl(r0, r0, 2)  # r0 = [0, 4, ..., 60]
    rotate(broadcast, r2, -A_ADDR)  # r5 = [A_ADDR] * 16
    add(r1, r0, r5)  # r1 = [A_ADDR, A_ADDR + 4, ..., A_ADDR + 60]
    rotate(broadcast, r2, -B_ADDR)  # r5 = [B_ADDR] * 16
    add(r3, r0, r5)  # r3 = [B_ADDR, B_ADDR + 4, ..., B_ADDR + 60]
    rotate(broadcast, r2, -C_ADDR)  # r5 = [C_ADDR] * 16
    add(rf1, r0, r5)  # rf1 = [C_ADDR, C_ADDR + 4, ..., C_ADDR + 60]

    # LOOP_NUMが0なら，つまりそれぞれのQPUが担当する要素数が16未満ならループに入らない．
    rotate(broadcast, r2, -LOOP_NUM)  # r5 = [LOOP_NUM] * 16
    mov(null, r5, cond="pushz")
    b(R.jmp, cond="anya")
    nop()
    nop()
    nop()

    # ループ処理
    with loop as iloop:
        # A + Bを計算してCに書き出す
        # A, B, Cのアドレスを1ループ分進める
        read_add_write()

        # LOOP_NUMを1つ減らして0になったらループを抜ける．つまりLOOP_NUM回繰り返す．
        rotate(broadcast, r2, -LOOP_NUM)
        sub(r0, r5, 1, cond="pushz")
        iloop.b(cond="anyna")

        # LOOP_NUMを1減らした値に更新
        eidx(r4)
        sub(null, r4, LOOP_NUM, cond="pushz")
        mov(r2, r0, cond="ifa")

    L.jmp

    # 最終ループ (SIMDの端数処理)
    rotate(broadcast, r2, -EDGE_MOD)  # r5 = [EDGE_MOD] * 16
    shl(r5, r5, 2)  # r5 = [EDGE_MOD * 4] * 16

    # A, B, Cのアドレスをそれぞれ端数文だけ戻すことで，SIMD幅の最後の要素と行列の最後の要素が一致するようにする．
    sub(r1, r1, r5)
    sub(r3, r3, r5)
    sub(rf1, rf1, r5)

    # A + Bを計算してCに書き出す
    # 戻した部分は再び計算することになるが仕方ない
    read_add_write()

    L.end

    # 終了処理
    exit_qpu()


def add(A: NDArray, B: NDArray) -> NDArray:
    SIMD_WIDTH = 16

    assert A.shape == B.shape

    n, m = A.shape

    if n * m <= 128:
        num_qpus = 1
    else:
        num_qpus = 8

    qpu_mod = (n * m) % num_qpus  # 要素数をQPUで等分したときの余り -> QPU7が処理
    proc_size = int((n * m) / num_qpus)  # QPU0~6それぞれが計算する要素数
    proc_size_lth = qpu_mod + proc_size  # QPU7が計算する要素数
    loop_num = int(proc_size / SIMD_WIDTH)  # QPU0~6のループ回数 (最後のループを除く)
    loop_num_lth = int(proc_size_lth / SIMD_WIDTH)  # QPU7のループ回数 (最後のループを除く)
    edge_mod = SIMD_WIDTH - proc_size % SIMD_WIDTH  # QPU0~6の最後のループの16要素のうち要素なし部分の個数
    edge_mod_lth = SIMD_WIDTH - proc_size_lth % SIMD_WIDTH  # QPU7の最後のループの16要素のうち要素なし部分の個数

    with Driver() as drv:
        # params setting
        A_ = drv.alloc((n, m), dtype="float32")
        B_ = drv.alloc((n, m), dtype="float32")
        C_ = drv.alloc((n, m), dtype="float32")
        A_[:] = A
        B_[:] = B

        # uniform setting
        unif = drv.alloc(16, dtype="uint32")
        unif[0] = A_.address
        unif[1] = B_.address
        unif[2] = C_.address
        unif[3] = proc_size
        unif[4] = loop_num
        unif[5] = edge_mod
        unif[6] = loop_num_lth
        unif[7] = edge_mod_lth
        code = drv.program(kernel, num_qpus=num_qpus)
        drv.execute(code, unif.addresses()[0], thread=num_qpus)

        return np.array(C_)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--rows", help="The number of rows", type=int, default=1024)
    parser.add_argument("--cols", help="The number of columns", type=int, default=1024)
    parser.add_argument("--iter", help="The number of iterations", type=int, default=10)
    args = parser.parse_args()
    n = args.rows
    m = args.cols
    iter = args.iter

    A = np.random.rand(n, m) * 0.1
    B = np.random.rand(n, m) * 0.1

    # Run the program
    cpu_time_sum = 0.0  # [ms]
    for i in range(iter):
        print(f"CPU Iteration {i + 1}")
        t_start = time.time()
        C_ref = A + B
        t_end = time.time()
        cpu_time_sum += (t_end - t_start) * 1000
    cpu_time_avg = cpu_time_sum / iter

    gpu_time_sum = 0.0  # [ms]
    for i in range(iter):
        print(f"GPU Iteration {i + 1}")
        t_start = time.time()
        C = add(A, B)
        t_end = time.time()
        gpu_time_sum += (t_end - t_start) * 1000
    gpu_time_avg = gpu_time_sum / iter

    print("cpu time: {} ms".format(cpu_time_avg))
    print("gpu time: {} ms".format(gpu_time_avg))
    print(C, C_ref)

    print("minimum absolute error: {:.4e}".format(float(np.min(np.abs(C_ref - C)))))
    print("maximum absolute error: {:.4e}".format(float(np.max(np.abs(C_ref - C)))))
    print("minimum relative error: {:.4e}".format(float(np.min(np.abs((C_ref - C) / C_ref)))))
    print("maximum relative error: {:.4e}".format(float(np.max(np.abs((C_ref - C) / C_ref)))))


if __name__ == "__main__":
    main()
