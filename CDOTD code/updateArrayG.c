#include "mex.h"
#include "math.h"

mwSize findOnePosition(double *inputArray, mwSize length) {
    mwSize i;
    for (i = 0; i < length; i++) {
        if (inputArray[i] == 1.0) {
            return i; //return i + 1;  // MATLAB的索引从1开始
        }
    }
    return 0;  // 如果没有找到1，返回0
}

void vectorMatrixMultiply(double *vector, double *matrix, double *result, mwSize matRows, mwSize matCols) {
    mwSize j,i;
    for (j = 0; j < matCols; j++) {
        result[j] = 0;
        for (i = 0; i < matRows; i++) {
            result[j] += vector[i] * matrix[i + j*matRows];
        }
    }
}

void findMaxValuePosition(double *inputArray, mwSize length, double *maxValue, mwSize *position) {
    *maxValue = inputArray[0];
    *position = 0;
    mwSize i;
    for (i = 1; i < length; i++) {
        if (inputArray[i] > *maxValue) {
            *maxValue = inputArray[i];
            *position = i;
        }
    }
    //(*position)++;  // MATLAB的索引从1开始
}

void vectorDotDivision(double *vector1, double *vector2, double *result, mwSize length) {
    mwSize i;
    for (i = 0; i < length; i++) {
        result[i] = vector1[i] / vector2[i];
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    double *inputArrayG;
    double *inputVectorN;
    double *inputVectorD;
    double *inputArrayU;
    double *inputArrayT;
    double *inputArrayUii;
    double *outputArrayG;
    double *outputVectorN;
    double *outputVectorD;
    double *outputArrayT;
    mwSize c, n, m;

    // 检查输入和输出参数的数量
    if (nrhs != 6) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nrhs", "Six inputs required.");
    }
    if (nlhs != 4) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:nlhs", "Four output required.");
    }

    // 获取输入
    inputArrayG = mxGetPr(prhs[0]);
    inputVectorN = mxGetPr(prhs[1]);
    inputVectorD = mxGetPr(prhs[2]);
    inputArrayU = mxGetPr(prhs[3]);
    inputArrayT = mxGetPr(prhs[4]);
    inputArrayUii = mxGetPr(prhs[5]);
    c = mxGetN(prhs[0]);
    n = mxGetM(prhs[0]);
    m = mxGetM(prhs[3]);

    // 创建输出向量
    plhs[0] = mxCreateDoubleMatrix((mwSize)n,(mwSize)c,mxREAL);
    outputArrayG = mxGetPr(plhs[0]);
    plhs[1] = mxCreateDoubleMatrix((mwSize)c,(mwSize)1,mxREAL);
    outputVectorN = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix((mwSize)c,(mwSize)1,mxREAL);
    outputVectorD = mxGetPr(plhs[2]);
    plhs[3] = mxCreateDoubleMatrix((mwSize)m,(mwSize)c,mxREAL);
    outputArrayT = mxGetPr(plhs[3]);

    // updateArrayG
    double *Gi = (double*)mxCalloc(c, sizeof(double));
    double maxValue;
    double *Ui = (double*)mxCalloc(n, sizeof(double));
    double *Ui2 = (double*)mxCalloc(m, sizeof(double));
    double *UiTk = (double*)mxCalloc(c, sizeof(double));
    double *Nk = (double*)mxCalloc(c, sizeof(double));
    double *Dk = (double*)mxCalloc(c, sizeof(double));
    double *N0 = (double*)mxCalloc(c, sizeof(double));
    double *D0 = (double*)mxCalloc(c, sizeof(double));
    double *result1 = (double*)mxCalloc(c, sizeof(double));
    double *result2 = (double*)mxCalloc(c, sizeof(double));
    double *delta = (double*)mxCalloc(c, sizeof(double));

    mwSize p, q;
    mwSize i,k;

    for (i = 0; i < n; i++) {
        for (k = 0; k < c; k++) {
            Gi[k] = inputArrayG[k*n+i];//G的第i行 i+0*n(行数) i+1*n
        }
        p = findOnePosition(Gi, c);
        if (inputVectorD[p]>1) {
            inputArrayG[p*n+i] = 0;  //G的第i行第p列
        }
        else {
            continue;
        }
        for (k = 0; k < m; k++) {
            Ui[k] = inputArrayU[i*m+k];//按列取 第i列 0+i*n（行数） 1+i*n
            Ui2[k] = 2*Ui[k];
        }
        vectorMatrixMultiply(Ui2, inputArrayT, UiTk, m, c);

        for (k = 0; k < c; k++) {
            if (k==p) {
                Nk[k] = inputVectorN[k];
                Dk[k] = inputVectorD[k];
                N0[k] = inputVectorN[k] - UiTk[k] + inputArrayUii[i];
                D0[k] = inputVectorD[k] - 1;
            }
            else {
                N0[k] = inputVectorN[k];
                D0[k] = inputVectorD[k];
                Nk[k] = inputVectorN[k] + UiTk[k] + inputArrayUii[i];
                Dk[k] = inputVectorD[k] + 1;
            }
        }
        vectorDotDivision(Nk, Dk, result1, c);
        vectorDotDivision(N0, D0, result2, c);

        for (k = 0; k < c; k++) {
            delta[k] = result1[k] - result2[k];
        }
        findMaxValuePosition(delta, c, &maxValue, &q);
        inputArrayG[q*n+i] = 1;  //改
        if (q!=p) {
            for (k = 0; k < m; k++) {
                inputArrayT[p*m+k] -= Ui[k];  //T的第p列 - Ui[k]
                inputArrayT[q*m+k] += Ui[k];
            }
            inputVectorN[p] = N0[p];
            inputVectorD[p] = D0[p];
            inputVectorN[q] = Nk[q];
            inputVectorD[q] = Dk[q];
        }
    }


    //返回结果
    for (i = 0; i < n*c; i++) {
        outputArrayG[i] = inputArrayG[i];
    }
    for (i = 0; i < c; i++) {
        outputVectorN[i] = inputVectorN[i];
    }
    for (i = 0; i < c; i++) {
        outputVectorD[i] = inputVectorD[i];
    }
    for (i = 0; i < m*c; i++) {
        outputArrayT[i] = inputArrayT[i];
    }

}


