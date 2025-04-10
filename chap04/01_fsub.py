#####################################################################
# ベクトル同士の割り算(割る数が2のべき乗の場合)
# 参考：https://github.com/Idein/py-videocore6/blob/master/examples/sgemm.py
#####################################################################
import numpy as np

from videocore6.assembler import qpu
from videocore6.driver import Driver


def exit_qpu():
    nop(sig=thrsw)
    nop(sig=thrsw)
    nop()
    nop()
    nop(sig=thrsw)
    nop()
    nop()
    nop()


@qpu
def kernel(asm):
    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(r0))
    nop(sig=ldunifrf(r1))
    nop(sig=ldunifrf(r3))

    # element_number
    eidx(r2)  # r2 = [0 ... 15]
    shl(r2, r2, 2)  # 各数値を4倍
    add(r0, r0, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成
    add(r1, r1, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmua, r0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r2))

    add(r0, r0, r3)
    mov(tmua, r0, sig=thrsw)
    nop()
    nop()
    nop(sig=ldtmu(r3))

    fsub(r2, r2, r3)

    mov(tmud, r2)  # 書き出すデータ
    mov(tmua, r1)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        # Input vectors
        A_ref = np.random.random(16).astype("float32")
        B_ref = np.random.random(16).astype("float32")

        # params setting
        inp = drv.alloc((2, 16), dtype="float32")
        out = drv.alloc(16, dtype="float32")

        inp[0][:] = A_ref
        inp[1][:] = B_ref

        # uniform setting
        unif = drv.alloc(3, dtype="uint32")
        unif[0] = inp.addresses()[0, 0]
        unif[1] = out.addresses()[0]
        unif[2] = 4 * 16

        # Run the program
        code = drv.program(kernel)
        drv.execute(code, unif.addresses()[0], thread=1)

        print(" a ".center(80, "="))
        print(A_ref)
        print(" b ".center(80, "="))
        print(B_ref)
        print(" a-b ".center(80, "="))
        print(out)
        print(" error ".center(80, "="))
        print(np.abs(A_ref - B_ref - out))


if __name__ == "__main__":
    main()
