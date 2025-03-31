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
def output_test(asm):
    # element_number
    mov(r0, 0)
    eidx(r2)  # r2 = [0 ... 15]

    # uniformから値を取り出す
    nop(sig=ldunifrf(rf0))
    sub(null, r2, 0, cond="pushz")
    mov(r0, rf0, cond="ifa")

    nop(sig=ldunifrf(rf0))
    sub(null, r2, 1, cond="pushz")
    mov(r0, rf0, cond="ifa")

    nop(sig=ldunifrf(rf0))
    sub(null, r2, 2, cond="pushz")
    mov(r0, rf0, cond="ifa")

    nop(sig=ldunifrf(rf0))
    sub(null, r2, 3, cond="pushz")
    mov(r0, rf0, cond="ifa")

    # アドレスを取り出す
    mov(r1, r0)
    eidx(r2)  # r2 = [0 ... 15]
    shl(r2, r2, 2)  # 各数値を4倍
    add(r1, r1, r2)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmud, r0)  # 書き出すデータ
    mov(tmua, r1)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        code = drv.program(output_test)

        result = drv.alloc(16, dtype="uint32")
        result[:] = 0

        unif = drv.alloc(4, dtype="uint32")
        unif[0] = result.addresses()[0]
        unif[1] = result.addresses()[0] + 1
        unif[2] = result.addresses()[0] + 2
        unif[3] = result.addresses()[0] + 3

        print("before")
        print(result)
        num_qpus = 1
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        print("after")
        print(result)


if __name__ == "__main__":
    main()
