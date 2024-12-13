#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
#include <omp.h>
#include <mpi.h>

using namespace std;

#define A_1 -4.
#define A_2 0.
#define B_1 4.
#define B_2 3.
#define M 40
#define N 40
#define delta 2e-7

double h1 = (B_1 - A_1) / M;
double h2 = (B_2 - A_2) / N;
double eps = pow(max(h1, h2), 2);

int blockM;
int blockN;//局部矩阵大小

///initialize all elements of solution(function) to zero as w^0
vector<double> w((M + 1)*(N + 1), 0);

void Split(int &m,int &n,int numProcs) {//numProcs:处理器数量
	m = M;
	n = N;
	if (numProcs == 1) {//单处理器情况，不分割
		return;
	}
	int procs = numProcs;
	while (procs > 1) {//等比例分割
		if (m >= n) {
			m /= 2;
		}
		else {
			n /= 2;
		}
		procs /= 2;
	}
}

double isInTriangle(double x, double y)
{
    double slope_AB = -2.0 / 3.0;
    double slope_BC = 2.0 / 3.0;
    if (y >= 0 && y <= slope_AB * (x - 3) && y <= slope_BC * (x + 3))
    {
        return 1;
    }
    else
    {
        return 0;
    }
}
//some operation in H space, which consists of the "inner" points of a grid function u on rectangle
//scalar product and corresponding norm in grid function space H
double funVScalProd(const vector<double> &u, const vector<double> &v, double h1, double h2) {
	double res = 0;
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
            //res += h1 * h2 * u[i][j] * v[i][j];
			res += h1 * h2 * u[i * (N + 1) + j] * v[i * (N + 1) + j];
		}
	}
	return res;
}

void funVScalProdMPI(double & res,const vector<double>& u, const vector<double>& v, const int* coords, double h1, double h2) {
	double resP = 0;//储存部分结果
	res = 0;//储存最终结果
#pragma omp parallel for collapse(2) reduction(+:res)
	for (int i = 0; i < blockM + 1; i++) {
		for (int j = 0; j < blockN + 1; j++) {//局部坐标 n0,n1
			int n0 = i + coords[0] * blockM, n1 = j + coords[1] * blockN;
			
			if (n1 == 0 || n1 == N || n0 == 0 || n0 == M) {
				continue;//边界检查
			}
			else if (i == 0 || j == 0) {
				continue;
			}
			else {
				resP += h1 * h2 * u[n0 * (N + 1) + n1] * v[n0 * (N + 1) + n1];
			}
		}
	} //每个进程都能得到结果总和（resp）
	MPI_Allreduce(&resP, &res, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);//全局归约操作
	//return res;
}

double funVNorm(const vector<double> &u, double h1, double h2) {
	return sqrt(funVScalProd(u, u, h1, h2));
}

void funVNormMPI(double & norm, const vector<double>& u, const int* coords, double h1, double h2) {
	funVScalProdMPI(norm, u, u, coords, h1, h2);
	norm = sqrt(norm);
}

auto funVSubtract(const vector<double> &u, const vector<double> &v) {
	vector<double> res((M + 1)*(N + 1), 0);
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {//跳过边界处理
            //res[i][j] = u[i][j] - v[i][j];
			res[i*  (N + 1) + j] = u[i * (N + 1) + j] - v[i * (N + 1) + j];
		}
	}
	return res;
}

void funVSubtractMPI(vector<double>& res, int * coords,const vector<double>& u, const vector <double>& v) {
	vector<double> resTemp((M + 1) * (N + 1), 0);
	//fill(res.begin(), res.end(), 0);
	#pragma omp parallel for collapse(2)
    //res 用于储存结果向量
    //resTemp 储存局部结果
	for (int i = 0; i < blockM + 1; i++) {
		for (int j = 0; j < blockN + 1; j++) {//为了遍历整个局部块
			int n0 = i + coords[0] * blockM, n1 = j + coords[1] * blockN;//n0表示行索引，n1表示列索引
			if (n1 == 0 || n1 == N || n0 == 0 || n0 == M) {
				continue;
			}
			else if (i == 0 || j == 0) {
				continue;
			}
			else {
				resTemp[n0 * (N + 1) + n1] = u[n0 * (N + 1) + n1] - v[n0 * (N + 1) + n1];
			}
		}
	}
	MPI_Allreduce(resTemp.data(), res.data(), (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}//将所有进程的局部结果restemp进行求和，归约到res中，resTemp.data(), res.data()提供了指向数组数据的指针

vector<double> funVmultConst(const vector<double> &u, double c) {
	vector<double> res((M + 1) * (N + 1), 0);
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i * (N + 1) + j] = u[i * (N + 1) + j] * c;
		}
	}
	return res;
}

void funVmultConstMPI(vector<double>& res, int* coords, const vector<double>& u, double c) {
	vector<double> resTemp((M + 1) * (N + 1), 0);
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM + 1; i++) {
		for (int j = 0; j < blockN + 1; j++) {
			int n0 = i + coords[0] * blockM, n1 = j + coords[1] * blockN;
			if (n1 == 0 || n1 == N || n0 == 0 || n0 == M) {
				continue;
			}
			else if (i == 0 || j == 0) {
				continue;
			}
			else {
				resTemp[n0 * (N + 1) + n1] = u[n0 * (N + 1) + n1] * c;
			}
		}
	}
	MPI_Allreduce(resTemp.data(), res.data(), (M + 1) * (N + 1), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
}

//prepration for calculation
// 
//define which area the 1/2node belongs to: 1 if inside D; remain 0 if outside D
void N_in_D(vector<int> &nodeType) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (isInTriangle(A_1 + (j + 0.5) * h1, A_2 + (i + 0.5) * h2) > 0) {
				nodeType[(i * N) + j] = 1;
			}
		}
	}
}

void N_in_DMPI(int rank,const int* coords, vector<int>& nodeType) {
	#pragma omp parallel for collapse(2)
	for (int i = 0; i < blockM; i++) {
		for (int j = 0; j < blockN; j++) {
			if (isInTriangle(A_1 +(blockN * coords[1] + j + 0.5) * h1, A_2 + (blockM * coords[0] + i + 0.5) * h2) > 0) {
				nodeType[(i + blockM * coords[0]) * N + j + blockN * coords[1]] = 1;
			}
		}
	}
}

//calculate the right side of equation Aw=B
void F_ij(vector<double> &Fij, const vector<int> &n) {//需要考虑边界情况
	#pragma omp parallel for
    for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			double Y_plus = A_2 + (i + 0.5) * h2;
            double Y_minus = A_2 + (i - 0.5) * h2;
            double X_plus = A_1 + (j + 0.5) * h1;
            double X_minus = A_1 + (j - 0.5) * h1;

            //int flag =0;
            // 全在虚域
            if (X_plus <= -3 || Y_plus <= 0 || Y_minus >= 2 || X_minus >= 3 || 2 * X_minus + 3 * Y_minus - 6 >= 0 || 3 * Y_minus - 2 * X_plus - 6 >= 0)
            {
                continue;
            }
            // 部分在D
            else if (n[(i - 1) * N + j] == 1 && n[i * N + j - 1] == 0){
                if(n[i * N + j] == 0 && n[(i - 1) * N + j - 1] == 0){
                    if (X_plus > 0){
                        if(Y_plus <= 2){
                            Fij[i * (N + 1) + j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * h2 * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                           // flag =1; cout << flag << " ";
                        }
                        else if (Y_plus > 2){
                            Fij[i * (N + 1) + j] = ((2 - Y_minus) * (-(3 * Y_minus / 2 - 3)) * 1 / 2 + (2 - Y_minus + (-2 * X_minus / 3 + 2) - Y_minus) * X_plus * 1 / 2) / (h1 * h2);
                           // flag =2; cout << flag << " ";
                        }
                    }
                    else if(X_plus <= 0) {
                        Fij[i * (N + 1) + j] = ((2 * X_plus / 3 + 2 - Y_minus) * (X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / (h1 * h2);
                       // flag =3; cout << flag << " ";
                    }
                }
                else if(n[i * N + j] == 1 && n[(i - 1) * N + j - 1] == 0){
                    Fij[i * (N + 1) + j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / h1;
                    //flag =4; cout << flag << " ";
                }
                else if(n[i * N + j] == 1 && n[(i - 1) * N + j - 1] == 1){
                    Fij[i * (N + 1) + j] = (h1 * h2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h1 * h2);
                    //flag =5; cout << flag << " ";
                }
            }
            else if (n[(i - 1) * N + j - 1] == 1 && n[i * N + j] == 0){
                if(n[i * N + j - 1] == 0 && n[(i - 1) * N + j] == 0){
                    if (X_minus < 0){
                        if(Y_plus <= 2){
                            Fij[i * (N + 1) + j] = ((-3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * h2 * 1 / 2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h1 * h2);
                            //flag =6; cout << flag << " ";
                        }
                        else{
                            Fij[i * (N + 1) + j] = ((2 - Y_minus + 2 * X_minus / 3 + 2 - Y_minus) * (-X_minus) * 1 / 2 + (2 - Y_minus) * (-3 * Y_minus / 2 + 3) * 1 / 2) / (h1 * h2);
                            //flag =7; cout << flag << " ";
                        }
                    }
                    else if(X_minus >= 0) {
                        Fij[i * (N + 1) + j] = ((-2 * X_minus / 3 + 2 - Y_minus) * (-3 * Y_minus / 2 + 3 - X_minus) * 1 / 2)/(h1 * h2);
                       // flag =8; cout << flag << " ";
                    }
                }
                else if (n[i * N + j - 1] == 1 && n[(i - 1) * N + j] == 0){
                    Fij[i * (N + 1) + j] = (( -3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * 1 / 2) / h1;
                    //flag =9; cout << flag << " ";
                }
                else if(n[i * N + j - 1] == 1 && n[(i - 1) * N +j] == 1){
                    Fij[i * (N + 1) + j] = (h1 * h2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                    
                }
            }
            else if (n[(i - 1) * N + j] == 0 && n[(i - 1) * N + j - 1] == 0 && n[i * N + j - 1] == 0 && n[i * N + j] == 0){
                if (Y_plus <= 2){
                    Fij[i * (N + 1) + j] = ((-3 * Y_plus / 2 + 3 - (3 * Y_plus / 2 - 3) + (-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3))) * 1 / 2) / h1;
                    
                }
                else {
                    Fij[i * (N + 1) + j] = ((-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3)) * (2 - Y_minus) * 1 / 2) / (h1 * h2);
                    
                }
            }
            else if (n[(i - 1) * N + j] == 1 && n[(i - 1) * N + j - 1] == 1){
                if (n[i * N + j - 1] == 0 && n[i * N + j] == 0){
                    if (Y_plus >= 2){
                        Fij[i * (N + 1) + j] = ((2 - Y_minus + (2 * X_minus / 3 + 2 - Y_minus)) * (-X_minus) * 1 / 2 + (2 - Y_minus + (-2 * X_plus / 3 + 2 - Y_minus)) * X_plus * 1 / 2) / (h1 * h2);
                       
                    }
                    else{
                        Fij[i * (N + 1) + j] = (h1 * h2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                       
                    }
                }
                else if(n[(i - 1) * N + j] == 1 && n[i * N + j] == 1){
                    Fij[i * (N + 1) + j] = 1;
                    
                }
            }
		}
	}
}

void F_ijMPI(vector<double> &Fij, const vector<int> &n) {//分成了有效区域块，只需要考虑有效部分
	#pragma omp parallel for
    for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			double Y_plus = A_2 + (i + 0.5) * h2;
            double Y_minus = A_2 + (i - 0.5) * h2;
            double X_plus = A_1 + (j + 0.5) * h1;
            double X_minus = A_1 + (j - 0.5) * h1;

           
            if (X_plus <= -3 || Y_plus <= 0 || Y_minus >= 2 || X_minus >= 3 || 2 * X_minus + 3 * Y_minus - 6 >= 0 || 3 * Y_minus - 2 * X_plus - 6 >= 0)
            {
                continue;
            }
            else if (n[(i - 1) * N + j] == 1 && n[i * N + j - 1] == 0){
                if(n[i * N + j] == 0 && n[(i - 1) * N + j - 1] == 0){
                    if (X_plus > 0){
                        if(Y_plus <= 2){
                            Fij[i * N  + j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * h2 * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                          
                        }
                        else if (Y_plus > 2){
                            Fij[i * N  + j] = ((2 - Y_minus) * (-(3 * Y_minus / 2 - 3)) * 1 / 2 + (2 - Y_minus + (-2 * X_minus / 3 + 2) - Y_minus) * X_plus * 1 / 2) / (h1 * h2);
                        }
                    }
                    else if(X_plus <= 0) {
                        Fij[i * N  + j] = ((2 * X_plus / 3 + 2 - Y_minus) * (X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / (h1 * h2);
                    }
                }
                else if(n[i * N + j] == 1 && n[(i - 1) * N + j - 1] == 0){
                    Fij[i * N  + j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / h1;
                }
                else if(n[i * N + j] == 1 && n[(i - 1) * N + j - 1] == 1){
                    Fij[i * N  + j] = (h1 * h2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h1 * h2);
                }
            }
            else if (n[(i - 1) * N + j - 1] == 1 && n[i * N + j] == 0){
                if(n[i * N + j - 1] == 0 && n[(i - 1) * N + j] == 0){
                    if (X_minus < 0){
                        if(Y_plus <= 2){
                            Fij[i * N  + j] = ((-3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * h2 * 1 / 2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h1 * h2);
                        }
                        else{
                            Fij[i * N  + j] = ((2 - Y_minus + 2 * X_minus / 3 + 2 - Y_minus) * (-X_minus) * 1 / 2 + (2 - Y_minus) * (-3 * Y_minus / 2 + 3) * 1 / 2) / (h1 * h2);
                        }
                    }
                    else if(X_minus >= 0) {
                        Fij[i * N  + j] = ((-2 * X_minus / 3 + 2 - Y_minus) * (-3 * Y_minus / 2 + 3 - X_minus) * 1 / 2)/(h1 * h2);
                    }
                }
                else if (n[i * N + j - 1] == 1 && n[(i - 1) * N + j] == 0){
                    Fij[i * N  + j] = (( -3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * 1 / 2) / h1;
                    //flag =9; cout << flag << " ";
                }
                else if(n[i * N + j - 1] == 1 && n[(i - 1) * N +j] == 1){
                    Fij[i * N  + j] = (h1 * h2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                }
            }
            else if (n[(i - 1) * N + j] == 0 && n[(i - 1) * N + j - 1] == 0 && n[i * N + j - 1] == 0 && n[i * N + j] == 0){
                if (Y_plus <= 2){
                    Fij[i * N  + j] = ((-3 * Y_plus / 2 + 3 - (3 * Y_plus / 2 - 3) + (-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3))) * 1 / 2) / h1;
                }
                else {
                    Fij[i * N  + j] = ((-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3)) * (2 - Y_minus) * 1 / 2) / (h1 * h2);
                }
            }
            else if (n[(i - 1) * N + j] == 1 && n[(i - 1) * N + j - 1] == 1){
                if (n[i * N + j - 1] == 0 && n[i * N + j] == 0){
                    if (Y_plus >= 2){
                        Fij[i * N  + j] = ((2 - Y_minus + (2 * X_minus / 3 + 2 - Y_minus)) * (-X_minus) * 1 / 2 + (2 - Y_minus + (-2 * X_plus / 3 + 2 - Y_minus)) * X_plus * 1 / 2) / (h1 * h2);
                    }
                    else{
                        Fij[i * N  + j] = (h1 * h2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h1 * h2);
                      
                    }
                }
                else if(n[(i - 1) * N + j] == 1 && n[i * N + j] == 1){
                    Fij[i * N  + j] = 1;
                   
                }
            }
		}
	}
}

//coefficients, determined by integral and will be calculated during the difference
//coefficient A
double a_ij(int i, int j, const vector<int>& nodeType) {
	double x = A_1 + (j - 0.5) * h1;
	double yU = A_2 + (i + 0.5) * h2;
	double yD = A_2 + (i - 0.5) * h2;
	int U = nodeType[i * N + j - 1];
	int D = nodeType[(i - 1) * N + j - 1];
	if (D + U == 0) {
		if (x < -3 || x > 3 || yU < 0 || yD > 2 || 2 * x + 3 * yD - 6 > 0 || 3 * yD - 2 * x - 6 > 0) {
			return 1 / eps;
		}
		else {
			return (-2. / 3 * abs(x) + 2) / h2 + (1 - (-2. / 3 * abs(x) + 2) / h2) / eps;
		}
	}
	else if (D + U == 1) {
		if (yD < 0) {
			return (yU) / h2 + (1 - yU / h2) / eps;
		}
		else {
			return (-2. / 3 * abs(x) + 2 - yD) / h2 + (1 - (-2. / 3 * abs(x) + 2 - yD) / h2) / eps;
		}
	}
	else {
		return 1;
	}
}

//coefficient B
double b_ij(int i, int j, const vector<int>& nodeType) {

	double y = A_2 + (i - 0.5) * h2;
	double xR = A_1 + (j + 0.5) * h1;
	double xL = A_1 + (j - 0.5) * h1;
	int R = nodeType[(i - 1) * N + j];
	int L = nodeType[(i - 1) * N + j - 1];
	if (L + R == 0) {
		if (y < 0 || y > 2 || xL > 3 || xR < -3 || 2 * xL + 3 * y - 6 > 0 || -2 * xR + 3 * y - 6 > 0) {
			return 1 / eps;
		}
		else {
			return (-3. / 2 * y + 3) * 2 / h1 + (1 - (-3. / 2 * y + 3) * 2/ h1) / eps;
		}
	}
	else if (L + R == 1) {
		if (xL < (3. / 2 * y - 3)) {
			return (xR - (3. / 2 * y - 3)) / h1 + (1 - (xR - (3. / 2 * y - 3)) / h1) / eps;
		}
		else {
			return (-3. / 2 * y + 3 - xL) / h1 + (1 - (-3. / 2 * y + 3 - xL) / h1) / eps;
		}
	}
	else {
		return 1;
	}
}

//perform A(w) and return the result, My w is defined in the entire rectangular space so that the subscripts of the internal points are the same as in the document.
auto operatorA(const vector<double> &w, const vector<int>& nodeType) {
	vector<double> res((M + 1) *(N + 1), 0);
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i * (N + 1) + j] = -(a_ij(i, j + 1, nodeType) * (w[i * (N + 1) + j + 1] - w[i * (N + 1) + j]) / h1 - a_ij(i, j, nodeType) * (w[i * (N + 1) + j] - w[i * (N + 1) + j - 1]) / h1) / (h1)-(b_ij(i + 1, j, nodeType) * (w[(i + 1) * (N + 1) + j] - w[i * (N + 1) + j]) / h2 - b_ij(i, j, nodeType) * (w[i * (N + 1) + j] - w[(i - 1) * N + j]) / h2) / (h2);
		}
	}
	return res;
}

void operatorAMPI(int rank,vector<double>& res ,const int* coords,const vector<double>& w, const vector<int>& nodeType) {
	vector<double> resTemp((M + 1) * (N + 1), 0);
    #pragma omp parallel for collapse(2)
    for (int i = 0 ; i < blockM + 1; i++) {
		for (int j = 0 ; j < blockN + 1; j++) {
			int n0 = i + coords[0] * blockM, n1 = j + coords[1] * blockN;
			int realCoords[2]{i+coords[0]*blockM,j+coords[1]*blockN};//计算并储存当前进程在全局网格中的实际坐标
			//coords[0]当前进程在全局网格中块的行坐标
			if (n1 == 0 || n1 == N|| n0 == 0 || n0 == M) {
				continue;
			}
			
			else if (i == 0 || j == 0) {
				continue;
			}
			else {
				resTemp[n0 * (N + 1) + n1] = -(a_ij(n0, n1 + 1, nodeType) * (w[n0 * (N + 1) + n1 + 1] - w[n0 * (N + 1) + n1]) / h1 - a_ij(n0, n1, nodeType) * (w[n0 * (N + 1) + n1] - w[n0 * (N + 1) + n1 - 1]) / h1) / (h1)-(b_ij(n0 + 1, n1, nodeType) * (w[(n0 + 1) * (N + 1) + n1] - w[n0 * (N + 1) + n1]) / h2 - b_ij(n0, n1, nodeType) * (w[n0 * (N + 1) + n1] - w[(n0 - 1) * (N + 1) + n1]) / h2) / (h2);
            }
        }
    }
    MPI_Allreduce(resTemp.data(),res.data(),(M+1)*(N+1),MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
    int provided;//储存线程支持级别
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);//初始化MPI环境
	if (provided != MPI_THREAD_MULTIPLE) {
		cout << "MPI do not Support" << endl;
		MPI_Abort(MPI_COMM_WORLD, 0);//如果调用这个函数，则所有参与的进程立即终止
	}
	
    int rank, numProcs;

	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);//获取当前进程ID，将其储存在rank中
    
	int dims[2]{ 0,0 };//储存每个维度中进程数量
	MPI_Dims_create(numProcs, 2, dims);//将numprods进程数分布在2D网格中，dims[0]是行，dims[1]是列
	int periods[2]{ 0,0 };//指示每个维度是否循环，0表示不循环
	MPI_Comm MPI_COMM_CART;//保存创建的笛卡尔拓扑
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, true, &MPI_COMM_CART);

	// Get my coordinates in the new communicator
	int cartCoords[2];//储存当前进程在新创建的笛卡尔拓扑中的坐标
	MPI_Cart_coords(MPI_COMM_CART, rank, 2, cartCoords);//调用MPI_Cart_coords函数，获取当前进程

	Split(blockM, blockN, numProcs);
    auto nodeType = vector<int>(M * N, 0);
	N_in_DMPI(rank, cartCoords, nodeType);
    vector<int>nodeTypeReduce(M * N, 0);
	MPI_Allreduce(nodeType.data(), nodeTypeReduce.data(), M * N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);//将所有进程的nodetype数组进行求和，将结果储存在nodeTypeReduce中
	auto Fij = vector<double>((M + 1) * (N + 1), 0);
	F_ij(Fij, nodeTypeReduce);
	int iterNum = 0;
	vector<double> wNewer((M + 1) * (N + 1), 0);
	double Arr, ArAr;
	vector<double> residual((M + 1) * (N + 1), 0);
	vector<double> Ar((M + 1) * (N + 1), 0);
	vector<double> Aw((M + 1) * (N + 1), 0);
	vector<double> tr((M + 1) * (N + 1), 0);
	vector<double> wErr((M + 1) * (N + 1), 0);
	double tau, error;
    MPI_Barrier(MPI_COMM_WORLD);//同步所有参与进程的函数

	double start = MPI_Wtime();
	do
	{
		w = wNewer;
        fill(wNewer.begin(), wNewer.end(), 0);//初始化
		fill(Aw.begin(), Aw.end(), 0);

		operatorAMPI(rank, Aw, cartCoords, w, nodeTypeReduce);
        
		fill(residual.begin(), residual.end(), 0);
		funVSubtractMPI(residual, cartCoords, Aw, Fij);
		fill(Ar.begin(), Ar.end(), 0);
		operatorAMPI(rank, Ar, cartCoords, residual, nodeTypeReduce);
		//funVScalProd(Ar, residual, h1, h2) / pow(funVNorm(Ar, h1, h2), 2);
		funVScalProdMPI(Arr, Ar, residual, cartCoords, h1, h2);
		funVScalProdMPI(ArAr, Ar, Ar, cartCoords, h1, h2);
		tau = Arr / ArAr;
		fill(tr.begin(), tr.end(), 0);
		funVmultConstMPI(tr, cartCoords, residual, tau);
		//wNewer = funVSubtract(w, funVmultConst(residual, tau));
		funVSubtractMPI(wNewer, cartCoords, w, tr);
		iterNum++;
		fill(wErr.begin(), wErr.end(), 0);
		funVSubtractMPI(wErr, cartCoords, wNewer, w);
		funVNormMPI(error, wErr, cartCoords, h_1, h_2);
		//cout  << "error: " << error <<endl;
	} while (error >= delta);
    MPI_Barrier(MPI_COMM_WORLD);
	//auto end = std::chrono::steady_clock::now();
    double end = MPI_Wtime();
	if (rank == 0) {
		cout << "Execution succeed!" << endl;
		cout << "Total iteration: " << iterNum << endl;
		cout << "Time elapsed: " <<(end - start) << "s" << endl;
    }
#pragma omp parallel{

#pragma omp single{
    if (rank == 0){
        ofstream fp("solution_mpi.txt");
        for (int i = 0; i < M + 1 ; i++) {
            for (int j = 0; j < N + 1 ; j++) {
                fp << setw(8) << setprecision(4) << wNewer[i * (N + 1) + j] << " ";//宽度为 8，保留 4 位小数
            }
            fp << endl;
        }
        fp.close();
    }
}
}
	//cout << "Execution succeed!"<<endl;
    //cout << "Total iteration: " << iterNum << endl;
    //cout << "Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
    MPI_Finalize();
    return 0;
	// for (int i = 0; i < M; i++)
    // {
    //     fp << A1 + i * h1 << " " << A2 << " " << 0.0 << endl;
    // }
    
    //Write interior points
    //     for (int i = 1; i < M ; i++){
    //        for (int j = 1; j < N ; j++){
    //         a << a_ij(i , j , n) ;
    //         if (j < N - 1){
    //             a << " ";
    //         }
    //     }
    //     a << endl;
    // }
    // a.close();
    // return 0;

    // for (int i = 1; i < M ; i++){
    //        for (int j = 1; j < N ; j++){
    //         b << b_ij(i , j , n) ;
    //         if (j < N - 1){
    //             b << " ";
    //         }
    //     }
    //     b << endl;
    // }
    // b.close();
}