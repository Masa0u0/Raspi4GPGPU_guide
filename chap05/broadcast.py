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
    # uniformから値を取り出す
    # uniformの読み取り位置はインクリメントされる(pop的動作)
    nop(sig=ldunifrf(r1))

    # element_number
    eidx(r0)  # r0 = [0 ... 15]
    nop()
    rotate(r2, r0, -3)  # 3つ左シフト: r2 = [3, ..., 15, 0, 1, 2]

    # broadcastは今まで通り"broadcast"という名前でも使えるが、
    # py-videocore6 では"r5rep"という名前でも使える
    # 検索やコードリーディングするときは要注意
    mov(broadcast, r2)  # r5 = [3, 3, ..., 3]

    # 出力ベクトルのアドレス配列を作成
    eidx(r3)  # r3 = [0 ... 15]
    shl(r3, r3, 2)  # 各数値を4倍
    add(r1, r1, r3)  # result[] のアドレスから ストライド=4バイトのアドレスベクトルを生成

    mov(tmud, r5)  # 書き出すデータ
    mov(tmua, r1)  # 書き出し先アドレスベクトル
    tmuwt()

    # GPUコードを終了する
    exit_qpu()


def main():
    with Driver() as drv:
        code = drv.program(output_test)

        result = drv.alloc(16, dtype="uint32")
        result[:] = 0

        unif = drv.alloc(1, dtype="uint32")
        unif[0] = result.addresses()[0]

        num_qpus = 1
        drv.execute(code, unif.addresses()[0], thread=num_qpus)
        print(result)


if __name__ == "__main__":
    main()
