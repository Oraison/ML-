# 03 : Linear Algebra - Review

## Matrices - overview
* 네모난 괄호 사이에 쓰여진 직사각형의 숫자 배열
  - 2D 배열
  - 보통 대문자로 이름을 짓는다(A, B, X, Y, J...)
* Matrix의 Dimention은 [Row $\times$ Columns]로 나타낸다
  - $R^{[r \times c]}$ 는 row의 크기가 r, column의 크기가 c인 Matrix를 의미한다
  $
  \\
  A=
  \begin{bmatrix}
  1401 & 191 \\
  1371 & 821 \\
  949 & 1437 \\
  147 & 1448 \\
  \end{bmatrix}
  $
  - 위의 Matrix는 $[ 4 \times 3]$ Matrix이다.
* Matrix Elements
  - $A_{(i,j)}$ = $i^{th}$ row, $j^{th}$ column을 의미한다.
   $
  \\
  A=
  \begin{bmatrix}
  1401 & 191 \\
  1371 & 821 \\
  949 & 1437 \\
  147 & 1448 \\
  \end{bmatrix}
  $
    + $A_{1,1}$ = 1401
    + $A_{1,2}$ = 191
    + $A_{3,2}$ = 1437
    + $A_{4,1}$ = 147

---

## Vectors = overview
* Vector는 n by 1 matrix로 표현한다.
  - 주로 소문자로 표현한다.
  - n개의 row, 1개의 column을 가지고 있다.
  $
  \\
  v=
  \begin{bmatrix}
  460 \\
  232 \\
  315 \\
  178 \\
  \end{bmatrix}
  $
  - 위는 4차원 백터이다.
* Vector Elements
  - $v_i$ = 백터의 $i^{th}$ element를 뜻한다.

---

## Matrix manipulation
### Addition
- 각각의 index의 원소를 서로 더한다.
- 같은 Dimension의 Matrix만 Addition 연산이 가능하다.
$
\\
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1 \\
\end{bmatrix}
\,+\,
\begin{bmatrix}
4 & 0.5 \\
2 & 5 \\
0 & 1 \\
\end{bmatrix}
\,=\,
\begin{bmatrix}
5 & 0.5 \\
4 & 10 \\
3 & 1 \\
\end{bmatrix}
$

### Muliplication by scalar
- Matrix의 각각 element에 scalar를 곱한다.
$
\\
3\, \times \,
\begin{bmatrix}
1 & 0 \\
2 & 5 \\
3 & 1 \\
\end{bmatrix}
\,=\,
\begin{bmatrix}
3 & 0 \\
6 & 15 \\
9 & 3 \\
\end{bmatrix}
$

### Division by a scalar
- 위의 Multiplication by scalar와 동일하다.
- 각각의 elements를 scalar로 나눈다

### combination of operands
- 곱셈, 나눗셈 먼저 순차적으로 계산한다.
$
\\
3 \times
\begin{bmatrix}
1 \\
4 \\
2 \\
\end{bmatrix} +
\begin{bmatrix}
0 \\
0 \\
5 \\
\end{bmatrix} - 
\begin{bmatrix}
3 \\
0 \\
6 \\
\end{bmatrix}
\div 3 =
\begin{bmatrix}
0 \\
12 \\
9
\end{bmatrix}
$

### Matrix by vector multiplication
- $[3 \times 2]$ Matrix $\times$ $[2 \times 1]$ vector
  - 연산 결과는 $[3 \times 1]$ matrix가 된다.
    + $[a \times b] \times [b \times c]$ 로 일반화 하면
      - 연산 결과는 $[a \times c]$ 가 된다.
$
\\
\begin{bmatrix}
1 & 3 \\
4 & 0 \\
2 & 1 \\
\end{bmatrix}
\times
\begin{bmatrix}
1 \\
5 \\
\end{bmatrix} = 
\begin{bmatrix}
1 \times 1 \;+\; 3 \times 5 \\
4 \times 1 \;+\; 0 \times 5 \\
2 \times 1 \;+\; 1 \times 5 \\
\end{bmatrix} = 
\begin{bmatrix}
16 \\
4 \\
7 \\
\end{bmatrix}
$
* $A \times x = y$
  - A = $[m \times n]$ Matrix
  - x = $[n \times 1]$ Vector
  - y = Result : m-dimentional Vector
* 실제 활용
  - 4개의 data set을 가정한다
  - $h_\theta(x) = -40\, +\, 0.25x$ 라는 함수도 가정한다
    + Matrix와 Vector의 곱으로 data를 표현할 수 있다
    + parameter를 data set matrix와 곱할 vector로 만든다
  - Prediction = Data Matrix $\times$ Parameters로 표현할 수 있다
$
\\
\begin{bmatrix}
1 &  2104 \\
1 &  1416 \\
1 &  1534 \\
1 &  852 \\
\end{bmatrix}
\times
\begin{bmatrix}
-40 \\
0.25
\end{bmatrix} = 
\begin{bmatrix}
-40 \times 1 \; + \; 0.25 \times 2104 \\
-40 \times 1 \; + \; 0.25 \times 1416 \\
-40 \times 1 \; + \; 0.25 \times 1534 \\
-40 \times 1 \, + \; 0.25 \times 852 \\
\end{bmatrix}
$
* data set에 $\theta_0$ 를 표현하기 위한 추가적인 1 column을 추가했다

### Matrix-Matrix multiplication
* General idea
  - Matrix-vector multiplication은 1개의 column만을 가지고 있다
  - 각각의 column을 vector라고 생각하고 각각의 vector에 대해 multiplication을 실시한다.
* Details
  - A $\times$ B = C
    + A = $[m \times n]$
    + B = $[n \times o]$
    + C = $[m \times o]$
      + o=1의 경우 B는 vector가 된다.
  - A의 column과 B의 row의 수가 같아야만 matrix muliplication이 가능하다.
* 예시
  - A $\times$ B
  $
  \\
  \begin{bmatrix}
  1 & 3 & 2 \\
  4 & 0 & 1 \\
  \end{bmatrix} \times
  \begin{bmatrix}
  1 & 3\\
  0 & 1\\
  5 & 2\\
  \end{bmatrix}
  $
  - B Matrix에서 첫 column vector를 가져와 계산한다.
  $
  \\
  \begin{bmatrix}
  1 & 3 & 2 \\
  4 & 0 & 1 \\
  \end{bmatrix} \times
  \begin{bmatrix}
  1 \\
  0 \\
  5 \\
  \end{bmatrix} = 
  \begin{bmatrix}
  11 \\
  9 \\
  \end{bmatrix}
  $
  - B Matrix에서 두번째 column vector를 가져와 계산한다.
  $
  \\
  \begin{bmatrix}
  1 & 3 & 2 \\
  4 & 0 & 1 \\
  \end{bmatrix} \times
  \begin{bmatrix}
  3 \\
  1 \\
  2 \\
  \end{bmatrix} = 
  \begin{bmatrix}
  10 \\
  14 \\
  \end{bmatrix}
  $
  - 결과
  $
  \\
  \begin{bmatrix}
  11 & 10 \\
  9  & 14 \\
  \end{bmatrix}
  $
  - $[2 \times 3]$ Matrix $\times$ $[3 \times 2]$ Matrix의 결과는 $[2 \times 2]$ Matrix이다.

---

## Implementation / use
* Hpuse price에서 3가지 가설을 설정할 수 있다.
* 모든 data에 대해서 3가지 가설을 확인하기 위해 Matrix-Matrix 곱셈이 효과적이다
    - Data patrix, Parameter matrix를 만들면 된다.
    - 예시
        + 4개의 집에 대해서 가격을 예측하고싶다
        + 3개의 가설을 설정한다
        $
        \\
        House\; Size:
        \begin{bmatrix}
        2104 \\
        1416 \\
        1534 \\
        852  \\
        \end{bmatrix}
        $
        
        $
        1: h_\theta(x) = -40 + 0.25x \\
        2: h_\theta(x) = 200 + 0.1x  \\
        3: h_\theta(x) = -150 + 0.4x \\
        $

        $
        \\
        \begin{bmatrix}
        1 &  2104 \\
        1 &  1416 \\
        1 &  1534 \\
        1 &  852  \\
        \end{bmatrix}
        \times
        \begin{bmatrix}
        -40  & 200 & -150 \\
        0.25 & 0.1 & 0.4  \\
        \end{bmatrix} = 
        \begin{bmatrix}
        486 & 410 & 692 \\
        314 & 34  & 416 \\
        344 & 353 & 464 \\
        173 & 285 & 191 \\
        \end{bmatrix}
        $


---

## Matrix multiplication properties
* 교환법칙
  - scalar와의 곱셈 연산에서는 성립한다.
  - Matrix에서는 성립하지 않는다
    + A $\times$ B $\neq$ B $\times$ A
* 결합법칙
    - $3 \times 5 \times 2 \; ==\; 3 \times 10 \;==\; 15 \times 2$
    - $A \times (B \times C) == (A \times B) \times C$
        + 결합법칙이 성립한다.
* Identity Matrix
    - 어떤 Matrix A와 Matrix I의 곱연산 결과가 A라면 I를 Identity Matrix라고 한다.
    - $I_{n \times n}$ 으로 표현하기도 한다.
    $
    \\
    \begin{bmatrix}
    1 & 0 \\
    0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 \\
    0 & 1 & 0 \\
    0 & 0 & 1 \\
    \end{bmatrix}
    \begin{bmatrix}
    1 & 0 & 0 & 0 \\
    0 & 1 & 0 & 0 \\
    0 & 0 & 1 & 0 \\
    0 & 0 & 0 & 1 \\
    \end{bmatrix}\\
    \;\, 2 \times 2 \qquad  3\times 3 \qquad\quad\;\,  4 \times 4
    $
    - Identity matrix의 성질
        + 대각선에 1로만 되어있다
        + 나머지는 0으로 채워져있다
        + A가 $[m \times n]$ 이라면
            + A $\times$ I (I : $[n \times n]$)
            + I $\times$ A (I : $[m \times m]$)
        + 이 된다
        + Matrix Mmultiplication에서 A $\times$ B $\neq$ B $\times$ A 이지만
        + B가 Identity Matrix라면 A $\times$ B == B $\times$ A 이다

---

## Inverse and Transpose Operations
* Matrix Inverse
    - Matrix A 에 대하여 $A^{-1}A$ = I가 되는 $A^{-1}$ 를 A의 Inverce라고 한다
        + Square Matrix만 Inverce가 존재한다
    - 예시
        + $[2 \times 2]$ Matrix
        $
        \\
        \begin{bmatrix}
        3 & 4 \\
        2 & 16 \\
        \end{bmatrix}
        \begin{bmatrix}
        0.4   & -0.1  \\
        -0.05 & 0.075 \\
        \end{bmatrix} = 
        \begin{bmatrix}
        1 & 0 \\
        0 & 1 \\
        \end{bmatrix}
        $
        + 어떻게 inverse를 계산하는가
            + inverse 계산은 쉽지 않다
            + Numerical software로 계산할 필요가 있다
    - 만약 A가 모두 0이라면 inverse matrix가 존재하지 않는다
## Matrix Transpose
* Matrix A($[m \times n]$)을 Transpose하면 같은 값을 가진 $[n \times m]$ Matrix가 된다.
    - row와 column을 바꾼다
* 어떻게 하는가
    - A Matirx의 첫 row를 $A^T$의 첫 column으로 한다.
    - A Matrix의 두번째 row를 ...
* A가 $[m \times n]$ Matrix일 때
    - B가 A의 tranpose라면
    - M는 $[n \times m]$ Matrix가 된다.
    - $A_{(i,j)} = B_{(j,i)}$
    $
    \\
    A = 
    \begin{bmatrix}
    1 & 2 & 0 \\
    3 & 5 & 9 \\
    \end{bmatrix}
    \; A^T = 
    \begin{bmatrix}
    1 & 3 \\
    2 & 5 \\
    0 & 9 \\
    \end{bmatrix}
    $