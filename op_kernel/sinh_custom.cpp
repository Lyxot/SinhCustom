#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;

class KernelSinh {
public:
    __aicore__ inline KernelSinh() {}
    __aicore__ inline void Init(/* 开发者填充参数列表 */GM_ADDR x, GM_ADDR y, uint32_t smallCoreDataNum,
                                uint32_t bigCoreDataNum, uint32_t finalBigTileNum, 
                                uint32_t finalSmallTileNum, uint32_t tileDataNum, 
                                uint32_t smallTailDataNum, uint32_t bigTailDataNum, 
                                uint32_t tailBlockNum)
    {
        //考生补充初始化代码
	    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
        uint32_t coreNum = AscendC::GetBlockIdx();
        uint32_t globalBufferIndex = bigCoreDataNum * AscendC::GetBlockIdx();
        this->tileDataNum = tileDataNum;
        if (coreNum < tailBlockNum) { 
          this->coreDataNum = bigCoreDataNum;
          this->tileNum = finalBigTileNum;
          this->tailDataNum = bigTailDataNum;
        }
        else { 
          this->coreDataNum = smallCoreDataNum;
          this->tileNum = finalSmallTileNum;
          this->tailDataNum = smallTailDataNum;
          globalBufferIndex -= (bigCoreDataNum - smallCoreDataNum) * (AscendC::GetBlockIdx() - tailBlockNum);
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + globalBufferIndex, this->coreDataNum);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + globalBufferIndex, this->coreDataNum);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileDataNum * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmp1, this->tileDataNum * sizeof(half));
        pipe.InitBuffer(tmp2, this->tileDataNum * sizeof(half));

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum;
        this->processDataNum = this->tileDataNum;
        for (int32_t i = 0; i < loopCount; i++) {
            if (i == this->tileNum - 1) {
              this->processDataNum = this->tailDataNum;
            }
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileDataNum], this->processDataNum);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // 算子实现
        if constexpr (std::is_same_v<DTYPE_X, int8_t>) {
            auto p1 = tmp1.Get<half>();
            auto p2 = tmp2.Get<half>();
            Cast(p1, xLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);
            Cast(p2, yLocal, AscendC::RoundMode::CAST_NONE, this->processDataNum);

            Exp(p2, p1, this->processDataNum);
            Reciprocal(p2, p1, this->processDataNum);
            Sub(p2, p1, p2, this->processDataNum);
            Muls(p2, p2, (half) 0.5, this->processDataNum);

            Cast(p1.ReinterpretCast<int16_t>(), p2, AscendC::RoundMode::CAST_RINT, this->processDataNum);
            ShiftLeft(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum); 
            ShiftRight(p1.ReinterpretCast<int16_t>(), p1.ReinterpretCast<int16_t>(), int16_t(8), this->processDataNum);
            Cast(p2, p1.ReinterpretCast<int16_t>(), AscendC::RoundMode::CAST_NONE, this->processDataNum);
            Cast(yLocal, p2, AscendC::RoundMode::CAST_NONE, this->processDataNum);
        }
        else {
            Exp(xLocal, xLocal, this->processDataNum);
            Reciprocal(yLocal, xLocal, this->processDataNum);
            Sub(yLocal, xLocal, yLocal, this->processDataNum);
            Muls(yLocal, yLocal, (half) 0.5, this->processDataNum);
        }
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileDataNum], yLocal, this->processDataNum);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmp1, tmp2;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    uint32_t coreDataNum;
    uint32_t tileNum;
    uint32_t tileDataNum;
    uint32_t tailDataNum;
    uint32_t processDataNum;
};

extern "C" __global__ __aicore__ void sinh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSinh op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.smallCoreDataNum, 
            tiling_data.bigCoreDataNum, tiling_data.finalBigTileNum, 
            tiling_data.finalSmallTileNum, tiling_data.tileDataNum, 
            tiling_data.smallTailDataNum, tiling_data.bigTailDataNum, 
            tiling_data.tailBlockNum);
    op.Process();
}
