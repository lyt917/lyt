#include <cmath>
#include <iostream>
#include <vector>
#include <iomanip>
#include <chrono>
#include <string>
#include <fstream>
using namespace std;

#define A_1 -4.
#define A_2 0.
#define B_1 4.
#define B_2 3.
#define M 160
#define N 180
#define delta 1e-7

double h_1 = (B_1 - A_1) / M;
double h_2 = (B_2 - A_2) / N;
double eps = pow(max(h_1, h_2), 2);

///initialize all elements of solution(function) to zero as w^0
vector<vector <double>> w(M + 1, vector<double>(N + 1, 0));

//F function y=-4/3x+4
double F(double x, double y)
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
double funVScalProd(const vector<vector<double> > &u, const vector<vector<double> > &v, double h_1, double h_2) {
	double res = 0;
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res += h_1 * h_2 * u[i][j] * v[i][j];
		}
	}
	return res;
}

double funVNorm(const vector<vector<double>> &u, double h_1, double h_2) {
	return sqrt(funVScalProd(u, u, h_1, h_2));
}

auto funVSubtract(const vector<vector<double>> &u, const vector<vector <double>> &v) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i][j] = u[i][j] - v[i][j];
		}
	}
	return res;
}

vector<vector<double>> funVmultConst(const vector<vector<double>> &u, double c) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i][j] = u[i][j] * c;
		}
	}
	return res;
}

//prepration for calculation
// 
//define which area the 1/2node belongs to: 1 if inside D; remain 0 if outside D
void nodeTypeDef(vector<vector<int>> &nodeType) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			if (F(A_1 + (j + 0.5) * h_1, A_2 + (i + 0.5) * h_2) > 0) {
				nodeType[i][j] = 1;
				//cout << 1;
			}
		}
	}
}

//calculate the right side of equation Aw=B
void FijDef(vector<vector<double>> &Fij, const vector<vector<int>> &n) {
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			double Y_plus = A_2 + (i + 0.5) * h_2;
            double Y_minus = A_2 + (i - 0.5) * h_2;
            double X_plus = A_1 + (j + 0.5) * h_1;
            double X_minus = A_1 + (j - 0.5) * h_1;

            //int flag =0;
            // 全在虚域
            if (X_plus <= -3 || Y_plus <= 0 || Y_minus >= 2 || X_minus >= 3 || 2 * X_minus + 3 * Y_minus - 6 >= 0 || 3 * Y_minus - 2 * X_plus - 6 >= 0)
            {
                continue;
            }
            // 部分在D
            else if (n[i - 1][j] == 1 && n[i][j - 1] == 0){
                if(n[i][j] == 0 && n[i - 1][j - 1] == 0){
                    if (X_plus > 0){
                        if(Y_plus <= 2){
                            Fij[i][j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * h_2 * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h_1 * h_2);
                           // flag =1; cout << flag << " ";
                        }
                        else if (Y_plus > 2){
                            Fij[i][j] = ((2 - Y_minus) * (-(3 * Y_minus / 2 - 3)) * 1 / 2 + (2 - Y_minus + (-2 * X_minus / 3 + 2) - Y_minus) * X_plus * 1 / 2) / (h_1 * h_2);
                           // flag =2; cout << flag << " ";
                        }
                    }
                    else if(X_plus <= 0) {
                        Fij[i][j] = ((2 * X_plus / 3 + 2 - Y_minus) * (X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / (h_1 * h_2);
                       // flag =3; cout << flag << " ";
                    }
                }
                else if(n[i][j] == 1 && n[i - 1][j - 1] == 0){
                    Fij[i][j] = ((X_plus - (3 * Y_plus / 2 - 3) + X_plus - (3 * Y_minus / 2 - 3)) * 1 / 2) / h_1;
                    //flag =4; cout << flag << " ";
                }
                else if(n[i][j] == 1 && n[i - 1][j - 1] == 1){
                    Fij[i][j] = (h_1 * h_2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h_1 * h_2);
                    //flag =5; cout << flag << " ";
                }
            }
            else if (n[i - 1][j - 1] == 1 && n[i][j] == 0){
                if(n[i][j - 1] == 0 && n[i - 1][j] == 0){
                    if (X_minus < 0){
                        if(Y_plus <= 2){
                            Fij[i][j] = ((-3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * h_2 * 1 / 2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2) / (h_1 * h_2);
                            //flag =6; cout << flag << " ";
                        }
                        else{
                            Fij[i][j] = ((2 - Y_minus + 2 * X_minus / 3 + 2 - Y_minus) * (-X_minus) * 1 / 2 + (2 - Y_minus) * (-3 * Y_minus / 2 + 3) * 1 / 2) / (h_1 * h_2);
                            //flag =7; cout << flag << " ";
                        }
                    }
                    else if(X_minus >= 0) {
                        Fij[i][j] = ((-2 * X_minus / 3 + 2 - Y_minus) * (-3 * Y_minus / 2 + 3 - X_minus) * 1 / 2)/(h_1 * h_2);
                       // flag =8; cout << flag << " ";
                    }
                }
                else if (n[i][j - 1] == 1 && n[i - 1][j] == 0){
                    Fij[i][j] = (( -3 * Y_plus / 2 + 3 - X_minus + (-3 * Y_minus / 2 + 3 - X_minus)) * 1 / 2) / h_1;
                    //flag =9; cout << flag << " ";
                }
                else if(n[i][j - 1] == 1 && n[i - 1][j] == 1){
                    Fij[i][j] = (h_1 * h_2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h_1 * h_2);
                    //flag =10; cout << flag << " ";
                }
            }
            else if (n[i - 1][j] == 0 && n[i - 1][j - 1] == 0 && n[i][j - 1] == 0 && n[i][j] == 0){
                if (Y_plus <= 2){
                    Fij[i][j] = ((-3 * Y_plus / 2 + 3 - (3 * Y_plus / 2 - 3) + (-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3))) * 1 / 2) / h_1;
                    //flag =11; cout << flag << " ";
                }
                else {
                    Fij[i][j] = ((-3 * Y_minus / 2 + 3 - (3 * Y_minus / 2 - 3)) * (2 - Y_minus) * 1 / 2) / (h_1 * h_2);
                    //flag =12; cout << flag << " ";
                }
            }
            else if (n[i - 1][j] == 1 && n[i - 1][j - 1] == 1){
                if (n[i][j - 1] == 0 && n[i][j] == 0){
                    if (Y_plus >= 2){
                        Fij[i][j] = ((2 - Y_minus + (2 * X_minus / 3 + 2 - Y_minus)) * (-X_minus) * 1 / 2 + (2 - Y_minus + (-2 * X_plus / 3 + 2 - Y_minus)) * X_plus * 1 / 2) / (h_1 * h_2);
                        //flag =13; cout << flag << " ";
                    }
                    else{
                        Fij[i][j] = (h_1 * h_2 - (3 * Y_plus / 2 - 3 - X_minus) * (Y_plus - (2 * X_minus / 3 + 2)) * 1 / 2 - (X_plus - (-3 * Y_plus / 2 + 3)) * (Y_plus - (-2 * X_plus / 3 + 2)) * 1 / 2) / (h_1 * h_2);
                        //flag =14; cout << flag << " ";
                    }
                }
                else if(n[i - 1][j] == 1 && n[i][j] == 1){
                    Fij[i][j] = 1;
                    //flag =15; cout << flag << " ";
                }
            }
		}
	}
}

//coefficients, determined by integral and will be calculated during the difference
//coefficient A
double coefA(int i, int j, const vector<vector<int>>& nodeType) {
	double x = A_1 + (j - 0.5) * h_1;
	double yU = A_2 + (i + 0.5) * h_2;
	double yD = A_2 + (i - 0.5) * h_2;
	int U = nodeType[i][j - 1];
	int D = nodeType[i - 1][j - 1];
	//if the two endpoints of the segment are outside d
	//if (nodeType[i - 1][j - 1] + nodeType[i][j - 1] == 0) {
	if (D + U == 0) {
		if (x < -3 || x > 3 || yU < 0 || yD > 2 || 2 * x + 3 * yD - 6 > 0 || 3 * yD - 2 * x - 6 > 0) {
			return 1 / eps;
		}
		else {
			return (-2. / 3 * abs(x) + 2) / h_2 + (1 - (-2. / 3 * abs(x) + 2) / h_2) / eps;
		}
	}
	//if one of the two endpoints of the segment is outside d
	//else if (nodeType[i - 1][j - 1] + nodeType[i][j - 1] == 1) {
	else if (D + U == 1) {
		if (yD < 0) {
			return (yU) / h_2 + (1 - yU / h_2) / eps;
		}
		else {
			return (-2. / 3 * abs(x) + 2 - yD) / h_2 + (1 - (-2. / 3 * abs(x) + 2 - yD) / h_2) / eps;
		}
	}
	else {
		return 1;
	}
}

//coefficient B
double coefB(int i, int j, const vector<vector<int>>& nodeType) {

	double y = A_2 + (i - 0.5) * h_2;
	double xR = A_1 + (j + 0.5) * h_1;
	double xL = A_1 + (j - 0.5) * h_1;
	int R = nodeType[i - 1][j];
	int L = nodeType[i - 1][j - 1];
	//if two endpoints of the segment are outside d
	//if (nodeType[i - 1][j - 1] + nodeType[i - 1][j] == 0) {
	if (L + R == 0) {
		if (y < 0 || y > 2 || xL > 3 || xR < -3 || 2 * xL + 3 * y - 6 > 0 || -2 * xR + 3 * y - 6 > 0) {
			return 1 / eps;
		}
		else {
			return (-3. / 2 * y + 3) * 2 / h_1 + (1 - (-3. / 2 * y + 3) * 2/ h_1) / eps;
		}
	}
	//if one of the two endpoints of the segment is outside d
	//else if (nodeType[i - 1][j - 1] + nodeType[i][j] == 1) {
	else if (L + R == 1) {
		if (xL < (3. / 2 * y - 3)) {
			return (xR - (3. / 2 * y - 3)) / h_1 + (1 - (xR - (3. / 2 * y - 3)) / h_1) / eps;
		}
		else {
			return (-3. / 2 * y + 3 - xL) / h_1 + (1 - (-3. / 2 * y + 3 - xL) / h_1) / eps;
		}
	}
	else {
		return 1;
	}
}

//perform A(w) and return the result, My w is defined in the entire rectangular space so that the subscripts of the internal points are the same as in the document.
auto operatorA(const vector<vector<double>> &w, const vector<vector<int>>& nodeType) {
	vector<vector<double>> res(M + 1, vector<double>(N + 1, 0));
	for (int i = 1; i < M; i++) {
		for (int j = 1; j < N; j++) {
			res[i][j] = -(coefA(i, j + 1, nodeType) * (w[i][j + 1] - w[i][j]) / h_1 - coefA(i, j, nodeType) * (w[i][j] - w[i][j - 1]) / h_1) / (h_1)-(coefB(i + 1, j, nodeType) * (w[i + 1][j] - w[i][j]) / h_2 - coefB(i, j, nodeType) * (w[i][j] - w[i - 1][j]) / h_2) / (h_2);
		}
	}
	return res;
}

int main() {
	auto nodeType = vector<vector<int>>(M, vector<int>(N, 0));
	auto Fij = vector<vector<double>>(M + 1, vector<double>(N + 1, 0));
	nodeTypeDef(nodeType);
	FijDef(Fij, nodeType);
	int iterNum = 0;
	vector<vector<double>> wNewer = w;
	double norm;
	auto residual = funVSubtract(operatorA(w, nodeType), Fij);
	auto Ar = operatorA(residual, nodeType);
	double tau = funVScalProd(Ar, residual, h_1, h_2) / pow(funVNorm(Ar, h_1, h_2), 2);
	//Generalized minimal residual method
	auto start = std::chrono::steady_clock::now();
	do
	{
		w = wNewer;
		residual = funVSubtract(operatorA(w, nodeType), Fij);
		Ar = operatorA(residual, nodeType);
		tau = funVScalProd(Ar, residual, h_1, h_2) / pow(funVNorm(Ar, h_1, h_2), 2);
		wNewer = funVSubtract(w, funVmultConst(residual, tau));
		iterNum++;
		if(iterNum % 2000 == 0){
			cout << "times" << iterNum << " " << funVNorm(funVSubtract(wNewer, w), h_1, h_2) << " ";
		}
	} while (funVNorm(funVSubtract(wNewer, w), h_1, h_2) >= delta);
	auto end = std::chrono::steady_clock::now();
	cout << "Execution succeed!"<<endl;
    cout << "Total iteration: " << iterNum << endl;
    cout << "Time : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << endl;
    ofstream fp("solution.txt");
    for (int i = 0; i < M + 1 ; i++) {
		for (int j = 0; j < N + 1 ; j++) {
			fp << setw(8) << setprecision(4) << wNewer[i][j] << " ";//宽度为 8，保留 4 位小数
        }
        fp << endl;
    }
    fp.close();
    return 0;
	///store result as .csv file and information about current execution as .txt file
	// string txtFileName("sequentialInfo"), csvFileName("data");
	// txtFileName += to_string(M) + " " + to_string(N) + ".txt";
	// csvFileName += to_string(M) + " " + to_string(N) + ".csv";
	// ofstream INFOFILE(txtFileName), CSVFILE(csvFileName);
	// INFOFILE << "Total iteration: " << iterNum << endl;
	// INFOFILE << "Time elapsed: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;
	// INFOFILE << "Solution: " << endl;
	// for (int i = M; i >= 0; i--) {
	// 	for (int j = 0; j < N + 1; j++) {
	// 		INFOFILE << setw(8) << setprecision(4) << wNewer[i][j] << " ";
	// 			if (j <= N - 1) {
	// 				CSVFILE << wNewer[M - i][j] << ",";
	// 			}
	// 			else {
	// 				CSVFILE << wNewer[M - i][j] << "\n";
	// 			}
	// 		}
	// 	INFOFILE << endl;
	// }
	// INFOFILE.close();
	// CSVFILE.close();
	// return 0;
}