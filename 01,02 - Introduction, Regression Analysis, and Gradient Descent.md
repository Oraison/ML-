# 01 and 02: Introduction, Regression Analysis, and Gradient Descent

## Introduction
* 무엇에 관련된 내용인가
 - 최신 기술
 - 어떻게 구현하는가
* ML은 무엇으로 구성되어있는가
  - search
  - Photo tagging
  - spam filters
* AI가 사람같은 지능을 갖도록 만든다
* 이 과정에는 뭐가 있는가
  - 최신 알고리즘을 공부한다
  - 알고리즘과 수학만으로는 힘들다
  - 실제 문제에서 어떻게 동작하는지에대한 노하우가 필요하다
* 왜 ML이 널리 쓰이는가
  - AI의 발달
  - Inteligent machine의 발명
    + 기계가 간단한 일을 하도록 프로그래밍 할 수 있다
      + 대부분의 경우 Hard-wiring AI는 너무 어렵다
    + 기계가 스스로 학습할 수 있는 방법을 마련하는게 가장 좋은 방법이다
      + 기계가 입력을통해 학습할 수 있다면
* 활용예시
  - Database mining
    + Web data
    + 의료기록
    + 생물학 자료
  - 직접 만들기 힘든 프로그램
    + 자동조정 헬리콥터
    + 필기 인식
    + 자연어처리
    + 컴퓨터 비전
  - 개인 맞춤 프로그램들

---

## ML이 뭘까
ML의 정의, 사용처
* 명확한 정의는 없다
  - 사람들이 ML을 정의하려고 한 것들은 있다
* ML의 구현방법
  - *Supervised learning*
    + 규칙을 기존에 정의한다
    + 이후 그를 실행하기 위한 새로운 지식을 발견한다
  - *Unsupervised learning*
    + 컴퓨터가 규칙을 배우게 한다
    + 이후 규칙을 이용해서 구조와 패턴을 결정한다
  - Reinforcement learning
  - Recommender systems

---

## Supervised learning
* 가장 많은 유형의 ML

### 예시
![Alt text](figs\fig1-1.png)
750 $feet^2$ size의 집이 있을 때 어느정도의 가격으로 측정되는가
1. Data를 관통하는 직선을 기준으로하면 $150,000정도라고 할 수 있다.
2. 주변의 Data를 기준으로 한다면 $200,000정도라고도 할 수 있다.
3. 직선이 아닌 곡선으로 Data를 나타내게 할 수도 있다.   
이들 모두 Supervised learning으로 구현할 수 있다.
* 이것의 의미
  - 알고리즘에 정답이 담긴 Data set을 제공했다.
  - 집의 실제 가격을 알고 있다
    + 우리는 training data를 통해 가격을 도출하는 것을 배울 수 있다
    + 이 알고리즘은 가격을 알지 못하는 새로운 training data에서 정답을 도출할 수 있다
* 이를 **regression problem**이라고 한다

* Supervised learning으로 "올바른" data를 얻을 수 있다
* Regression problem, Classification problem이 이에 속한다.

---

## Unsupervised learning
* 그 다음으로 많은 문제 유형

* Unsupervised learning은 labeling되지 않은 data를 기반으로 한다.
  - data set을 주고 구조를 파악할 수 있는지 확인
* 이를 구현하는 방법은 데이터를 그룹으로 나누는 것이다.
  - 이게 **Clustering algorithm**이다

### Clustering algorithm

* Clustering algorithm 예시
  - Google news
    + 비슷한 기사끼리 그룹으로 묶는다
  - Genomics
  - Microarray data(유전자 배열)
  - computer cluster 구성
    + 잠재적인 취약점 분석 및 효율적인 작업 분배
  - 천문학 자료 분석
* Basically 
  - Structure을 자동으로 만들어낼 수 있는가
  - 이를 명시적으로 제공하지 않았으므로 **Unsupervised learning**이다

---

## Linear Regression
* Housing pirce에서 이미 활용해봤다
* 필요한것들
  - Training set(data set)
  - Notation
    + m : # of traning examples
    + x's : input variables / features
    + y's : output variables / 'target' variables
      + (x, y) : single training example
      + ($x^{(i)}$, $y^{(i)}$) : i번째 training set
* training set으로 무엇을 하는가
  - Training set을 선택
  - Learning algorithm을 실행
  - Algorithm outputs a function($h_\theta$)   
  $h_\theta(x) = \theta_0 + \theta_1x$
    + $\theta_0$ : zero condition
    + $\theta_1$ : gradient
  - 변수가 1개인 linear regression이므로 **univariate linear regression**이라고 한다

### Linear regression - implementation 
* Data를 가장 잘 나타내는 직선을 그리는 함수를 만들어야한다.
* $\theta_i$의 값을 찾는다
  - 값이 달라지면 다른 함수가 생긴다
* training set을 기반으로 parameters를 만들어야한다
  - $h_\theta(x)$와 $y$가 비슷하도록 parameters를 설정해야한다.
* 일반화하면
  - minimization problem을 해결하는 것과 같다.
  - Minimize $(h_\theta(x) - y)^2$
    + 모든 x와 y에 대해서 최소화되게 해야한다.
  - 모든 training set에 대하여 위의 식을합산한 값의 최소값
    + $\displaystyle\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^i) - y^i)^2$
      + $\frac{1}{2m}$의 의미
        + $\frac{1}{m}$ : 차이값의 평균
        + $\frac{1}{2m}$ : $\frac{1}{2}$를 해서 계산이 조금 더 쉬워진다.   
          + $\frac{1}{2}$를 해도 이를 최소화하는 $\theta$의 값은 변하지 않는다.
* 이를 함수 수식으로 정리하면
  $\displaystyle J(\theta_0,\theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2$ 가 된다

---

## Gradient descent algorithm
* J의 최소값 찾기
* Gradient descent란
  - ML 전반적으로 적용되는 최소화 기법
* $J(\theta_0,\, \theta_1)$의 최소값은 찾기 쉽다.
* $J(\theta_0,\, \theta_1,\, \dots\, ,\, \theta_n)$의 경우는 최소값을 바로 찾기 힘들다

### 동작원리
* 특정 값을 선택한다
  - ex) (0,0)
  - $\theta_0$와 $\theta_1$의 값을 조금씩 바꿔가며 $J(\theta_0,\, \theta_1)$을 풀어나간다
- parameter를 변경할 때 $J(\theta_0,\, \theta_1)$가 최소화되도록 선택한다
- local minimum으로 수렴할 때 까지 반복한다
- 시작지점에 따라 도달하는 최소값이 달라진다는 특성이 있다

### A more formal definition
* $\displaystyle \theta_j := \theta_j - \alpha\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$(j=0 and j=1)
* 식의 의미
  - $\theta_j$에 대하여 J를 편미분한 값과 $\alpha$를 곱하여 $\theta_j$를 업데이트한다
  - $\alpha$ : learning rate
    + $\theta_j$를 얼마나 변화시킬 것인가
    + $\alpha$가 너무 작으면 $\theta_j$의 변화량이 너무 작아 너무 오래 걸린다
    + $\alpha$가 너무 크면 최소값을 초과하여 실패할 수 있다
  - $\displaystyle \frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$
    + j=0, j=1일때를 동시에 계산해서 동시에 update해야한다
    + temp를 두고 temp를 계산해서 동시에 $\theta_0,\,\theta_1$을 동시에 update한다
    + 접점의 기울기가 양수일 때(증가하고 있을 때) $\theta_j$를 작은 값으로 update한다
    + 접점의 기울기가 음수일 때(감소하고 있을 때) $\theta_j$를 큰 값으로 update한다
  - local minimum에 도달하면
    + Gradiant가 0이 된다
    + derivative term이 0이된다
    + $\theta_j = \theta_j - 0$가 된다
    + $\theta_j$가 유지된다

---

## Linear regression with gradient descent
$\displaystyle \frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1)$
* 위의 식을 전개해보자
  - $\displaystyle J(\theta_0,\, \theta_1)\, = \, \frac{1}{2m}$ 이고
  - $\displaystyle h_\theta(x)\, =\, \theta_0 + \theta_1\times x$ 이다.
* 이를 적용하면 식이   
$
\begin{aligned}
\displaystyle &\frac{\partial}{\partial\theta_j}J(\theta_0,\theta_1) \\
=& \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 \\
=& \frac{\partial}{\partial\theta_j}\frac{1}{2m}\sum_{i=1}^m(\theta_0 + \theta_1x^{(i)} - y^{(i)})^2
\end{aligned} 
$   
   위와같은 형태로 나온다
* 이를 $j=0,\, j=1$ 에 대하여 미분하면  
$
\displaystyle
j=0:\,\frac{\partial}{\partial\theta_0}J(\theta_0,\,\theta_1)\, =\, \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)})\\
j=1:\,\frac{\partial}{\partial\theta_1}J(\theta_0,\,\theta_1)\, =\, \frac{1}{m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{(i)}) \cdot x^{(i)}
$   
  위와같은 형태가 된다

* 위의 식을 풀기 위해서는 multivariate calculus가 필요하다
  - 이 식을 다시 gradient desent algorithm에 적용한다
* 동작원리
  - 초기값에 따라 서로 다른 local optimum을 갖게될 수 있다
  - linear regression cost function은 항상 convex function이다
    + 항상 하나의 minimum 값을 갖는다
      - Bowl shaped
      - 1개의 Global obtima
        * gradient desent가 항상 global optima로 수렴한다
* 이는 Batch Gradient Desent이다
  - 매번 모든 training data를 참조한다
    + 매번 m개의 training sample을 계산한다
  - 적은 양의 data subset에 대해서는 not-batch version도 있다.
* minimum fucntion의 solution을 찾을 수 있는 numerical solution이 있다
  - Normal equations method
  - Gradient descent는 많은 data set에서 확장하기 용이하다
  - 다양한 상황과 ML에서 사용된다

---

## Important Extensions

### 1. Normal equation for numeric solution
  * minimization problem을 numeric method를 이용하면 gradient descent를 사용하므로써 iterative approach를 하지 않고 정확히 min $J(\theta_0,\, \theta_1)$ 를 풀 수 있다
  * Normal equations method
    - 장점
      + $\alpha$ 가 필요없다
      + 몇몇 문제는 더 빨리 풀 수 있다
    - 단점
      + 조금 더 복잡하다
### 2. 여러가지 특성
  * 가격에 대해 많은 parameters가 있을 수 있다
    - 예를 들면
      + 집의 크기
      + 연식
      + 방의 수
      + 층수
    - 각각을 x1, x2, x3, x4라고 할 때
  * 여러 변수가 있으면 각각을 연관짓기 어렵게 된다
    - 3차원 이상은 그래프로 표현할 수 없다
    - 표기하는 것도 힘들다
      + Linear algebra(Matrix)로 표현하는 것이 제일 좋은 방법이다
      + matrix와 vextor들로 표현하면 된다

  $
  X=
  \begin{bmatrix}
  2104 & 5 & 1 & 45 \\
  1416 & 3 & 2 & 40 \\
  1534 & 3 & 2 & 30 \\
  852  & 2 & 1 & 36 \\
  \end{bmatrix}
  \;\;\;
  y=
  \begin{bmatrix}
  460 \\
  232 \\
  315 \\
  172 \\
  \end{bmatrix}
  $

  * 위의 matrix에는
    - 집의 크기
    - 방의 수
    - 층수
    - 연식   
  에대한 정보가 있다
  * Matrix의 곱으로 표현하면
  
  $
  \displaystyle 
  \begin{aligned}
  \overrightarrow{y}\, =&\; X \times\overrightarrow{\theta} \\
  \begin{bmatrix}
  460 \\
  232 \\
  315 \\
  172 \\
  \end{bmatrix}
  \; =& \; 
  \begin{bmatrix}
  2104 & 5 & 1 & 45 \\
  1416 & 3 & 2 & 40 \\
  1534 & 3 & 2 & 30 \\
  852  & 2 & 1 & 36 \\
  \end{bmatrix}
  \times
  \begin{bmatrix}
  집의 크기 \\
  방의 수 \\
  층수 \\
  연식 \\
  \end{bmatrix}
  \end{aligned}
  $

  위와 같이 표현할 수 있다

  * Linear algibra로 복잡한 linear regression을 효과적으로 표현할 수 있다
    - 많은 data set을 다루기 위한 좋은 방법
    - 백터로 표현하는 것은 최적화 기술의 일반적인 방법이다