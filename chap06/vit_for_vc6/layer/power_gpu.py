import onnx
import numpy as np
from .layer_base import Eltwise_Layer
from videocore6.assembler import qpu
from videocore6.driver import Driver


class Pow_gpu(Eltwise_Layer):
    def __init__(self, model, node, tensor_data, use_gpu=False, drv=None):
        super().__init__(model, node, tensor_data, use_gpu, drv)
        self.set_1input(kernel)

    def run(self):
        if self.tensor_data[self.input_name[1]] == 2.0:
            self.drv.execute(self.code, self.unif.addresses()[0], thread=self.num_qpus)
        else:
            print("Warning: The GPU's pow function only supports Y==2. This layer will be executed on the CPU")
            input_data1 = self.tensor_data[self.input_name[0]]
            input_data2 = self.tensor_data[self.input_name[1]]
            output = np.power(input_data1, input_data2)
            if self.use_gpu:
                self.tensor_data[self.output_name][:] = output
            else:
                self.tensor_data[self.output_name] = output


def get_thid():
    tidx(r0)
    shr(r0, r0, 2)
    band(r0, r0, 0b1111)


def read_add_write():
    mov(tmua, r1, sig=thrsw)
    mov(r0, 1)  # nop()
    shl(r0, r0, 6)  # nop()
    nop(sig=ldtmu(rf5))
    add(r1, r1, r0)  # nop()
    fmul(rf0, rf5, rf5)
    mov(tmud, rf0)
    mov(tmua, rf1)
    add(rf1, rf1, r0)
    tmuwt()


@qpu
def kernel(asm, num_qpus):
    A_ADDR = 0
    C_ADDR = 1
    PSIZE = 2
    LOOP_NUM = 3
    EDGE_MOD = 4
    LOOP_NUM_LTH = 5
    EDGE_MOD_LTH = 6
    eidx(r0).mov(r2, 0)
    for idx in [A_ADDR, C_ADDR, PSIZE, LOOP_NUM, EDGE_MOD, LOOP_NUM_LTH, EDGE_MOD_LTH]:
        nop(sig=ldunifrf(r5))
        sub(null, r0, idx, cond="pushz")
        mov(r2, r5, cond="ifa")

    if num_qpus == 1:
        mov(r0, 0)
    elif num_qpus == 8:
        get_thid()  # r0にthread idを格納
    else:
        raise Exception("num_qpus must be 1 or 8")

    for i in range(64):
        mov(rf[i], 0.0)

    sub(null, r0, 7, cond="pushz")
    b(R.not_th_id7, cond="anyna")
    nop()
    nop()
    nop()
    eidx(r3)

    rotate(broadcast, r2, -LOOP_NUM_LTH)
    sub(null, r3, LOOP_NUM, cond="pushz")
    mov(r2, r5, cond="ifa")

    rotate(broadcast, r2, -EDGE_MOD_LTH)
    sub(null, r3, EDGE_MOD, cond="pushz")
    mov(r2, r5, cond="ifa")

    L.not_th_id7
    # set per thread data
    rotate(broadcast, r2, -PSIZE)
    umul24(r1, r5, r0)
    shl(r1, r1, 2)
    nop()
    nop()
    eidx(r4)
    sub(null, r4, A_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")
    sub(null, r4, C_ADDR, cond="pushz")
    add(r2, r2, r1, cond="ifa")

    eidx(r0)
    shl(r0, r0, 2)
    rotate(broadcast, r2, -A_ADDR)
    add(r1, r0, r5)
    rotate(broadcast, r2, -C_ADDR)
    add(rf1, r0, r5)

    rotate(broadcast, r2, -LOOP_NUM)
    mov(null, r5, cond="pushz")
    b(R.jmp, cond="anya")
    nop()
    nop()
    nop()

    with loop as iloop:
        read_add_write()
        rotate(broadcast, r2, -LOOP_NUM)
        sub(r0, r5, 1, cond="pushz")
        iloop.b(cond="anyna")
        eidx(r4)
        sub(null, r4, LOOP_NUM, cond="pushz")
        mov(r2, r0, cond="ifa")
    L.jmp

    # SIMDの端数処理
    rotate(broadcast, r2, -EDGE_MOD)
    shl(r5, r5, 2)
    sub(r1, r1, r5)
    sub(r3, r3, r5)
    sub(rf1, rf1, r5)
    read_add_write()

    L.end
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
