# Support Vector Machines(SVMs)

## Support Vector Machine(SVM) - Optimization objective

* 이전까지 성능이 크게 차이나지 않는 supervised learning 알고리즘을 배웠다
    - 이 알고리즘에서 중요한 것은
        + 많은 양의 training data
        + 알고리즘을 적용할 기술
* Support vectyor machin(SVM)은 많이 사용되는 supervised learning algorithm이다
    - SVM은 종종 non-linear function을 학습할 때 logistic regression과 neural networks보다 깔끔한 방법이기도 하다

### An alternative view of logistic regression
* Logistic regression을 변형하여 SVM을 생성할 것이다
    - 이전의 logistic regression은 아래와 같은 형태이다
    $\\\displaystyle h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$
    - Sigmoid activation function은 아래와 같다
    ![Alt text](figs\fig12-1.png)
    - z는 그림 아래와 같이 $(\theta^Tx)$ 로 정의했다
* Logistic regression이 왜 필요한가
    - y=1일 때
        + $h_\theta(x)$ 가 1에 근접하게 만들고 싶다
        + $h_\theta(x)$ 가 1에 근접할 때 $(\theta^Tx)$ 가 0보다 매우 커야한다
    - y=0일 때
        + $h_\theta(x)$ 가 0에 근접하게 만들고 싶다
        + $h_\theta(x)$ 가 0에 근접할 때 $(\theta^Tx)$ 가 0보다 매우 작아야한다
* Logistic regression
    - Cost function은 아래와 같다
    $\\\displaystyle 
    -(y \log h_\theta(x) + (1-y) \log(1-h_\theta(x)))
    $
        + 모든 x에 대한 cost function은 각각의 training set의 x와 y를 통해 얻은 결과의 총합을 m으로 나눈 형태이다
* 위의 cost function의 $h_\theta(x)$ 를 대입한다면 아래와 같다
$\\\displaystyle
-(y \log \frac{1}{1+e^{-\theta^Tx}} + (1-y) \log(1-\frac{1}{1+e^{-\theta^Tx}}))
$

* y=1일 때의 z에대한 cost function의 그래프이다</br>
![Alt text](figs\fig12-2.png)
    - z가 크면 cost는 작아진다
    - z가 0이거나 음수가 되면 cost가 매우 커진다
* y=0일 때의 z에대한 cost function의 그래프이다</br>
![Alt text](figs\fig12-3.png)
    - z가 음수면 cost는 작아진다
    - z가 커지면 되면 cost가 매우 커진다

### SVM cost functions from logistic regression cost functions

* SVM을 만들기 위해서는 cost function을 재정의 할 필요가 있다
    - y=1
        + 기존의 logistic regression의 cost function은 곡선으로 만들어졌다
            + SVM의 cost function은 2개의 직선(분홍색)으로 이루어진다
            ![Alt text](figs\fig12-5.png)
                - z=1이 기준이다
                    * 1부터는 평평하다(기울기가 0이다)
                    * 1보다 작아지면 값이 증가한다
                    * 평평할 때 cost가 0이 된다
            + 이것이 새로운 y=1의 cost fucntion이다
                - 이 함수는 optimization problem을 연산할 때 더 쉽게 계산할 수 있다는 이점이 있다
                - 이를 $cost_1(z)$ 라고 한다
* y=0
    - 같은 방식으로 y=0일 때의 cost function을 표현한다
    ![Alt text](figs\fig12-6.png)
        + 이를 $cost_0(z)$ 라고 한다
* SVM의 cost function을 그래프로 그렸다
    - 이를 어떻게 구현할 것인가

### The complete SVM cost function

* Logistic regression은 아래와 같다
![Alt text](figs\fig12-4.png)
    - 위의 식에는 -log가 있어 좋지 않아보인다
* SVM에서는 logistic regression의 y=1, y=0의 항을 다음과 같이 쓴다
    - $cost_1(\theta^Tx)$
    - $cost_0(\theta^Tx)$
* 이를 적용하면 다음과 같은 식을 얻을 수 있다
![Alt text](figs\fig12-7.png)

### SVM notation is slightly different

* 이를 SVM으로 변경하기 위해서는 몇몇가지를 바꿔야한다
1. $\frac{1}{m}$ 제거
    - 최소값을 찾기 위한 것이다
    - 최소값에 특정한 상수를 곱하더라도 최소값은 유지된다
    - 상수 $\frac{1}{m}$ 이 있을 때의 최소값은 $\frac{1}{m}$ 이 없어도 최소값이다
2. Logistic regression의 두 항을 변경한다
    - 위의 식은 두개의 항으로 되어있다
        + Training data set term($\sum$ i = 1 to m) : A
        + Regularization term($\sum$ i = 1 to n) : B
    - 위의 식은 $A + \lambda B$ 의 형태를 갖고있다
    - $\lambda$ 는 두 항 중 어떤 항에 더 가중치를 줄지를 결정하기 위한 상수이다
    - SVM에서는 $\lambda$ 대신 C를 사용할 것이다
        + CA + B
        + $C = \frac{1}{\lambda}$ 라고 볼 수 있다
            + $CA + B$ 와 $A + \lambda B$ 는 같은 결과를 만든다
* 위의 두가지를 변경하면 아래와 같은 식으로 표현할 수 있다
![Alt text](figs\fig12-8.png)
* Logistic과는 달리 $h_\theta(x)$ 가 확률을 예측하지는 않지만, 1과 0으로 직접 예측한다
    - $\theta^Tx \geq 0$ 이면 $h_\theta(x) = 1$
    - 그렇지 않으면 $h_\theta(x) = 0$

---

## Large margin intuition

* SVM을 large margin classifiers라고 부르기도 한다
    - large margin classifiers가 어떤 의미인지, SVM hypothesis가 어떻게 생겼는지를 생각해볼 필요가 있다
    - 위에서 살펴본 SVM cost function은 아래와 같이 나타낼 수 있다
    ![Alt text](figs\fig12-9.png)
    - 왼쪽은 $cost_1$ 이고, 오른쪽은 $cost_0$ 이다
    - 언제 cost가 0이 되는가
        + y=1일 때
            + $cost_1(z) = 0$
                - $z \geq 1$
        + y=0일 때
            + $cost_0(z) = 0$
                - $z \leq -1$
    - SVM의 톡특한 특징
        + Positive example의 경우 z가 0 이상이길 바란다
            + 1로 예측한 경우
        + SVM의 경우 0이 아니라 0보다 조금 더 큰 수를 원한다
            + 추가적인 margin factor를 가진다
* Logistic regression도 비슷한 동작이 가능하다
    - C가 매우 클 경우
        + C = 100,000
        + CA + B 를 최소화 하는 경우를 생각해보자
            + C가 매우 크면 A를 0에 가깝게 만들어야된다
            + 어떻게 A를 0으로 만들 수 있을까
        + A=0 을 만드는 법
            + y=1일 때
                - A 항을 0으로 만들기 위한 $\theta$ 를 찾아야 한다
                    * $\theta^Tx$ 가 1 이상일 때
            + y=0일 때
                - A 항을 0으로 만들기 위한 $\theta$ 를 찾아야 한다
                    * $\theta^Tx$ 가 -1 이하일 때
        + A=0이 보장된다면 B항만 minimizing하면 된다
            - A=0이면 A*C = 0이기 때문
        + B의 minimization은 아래와 같다</br>
        ![Alt text](figs\fig12-10.png)
    - 이를 계산하면 흥미로운 decision bondaries를 얻을 수 있다
    ![SVM decision boundary with logistic regression's](figs\fig12-11.png)
    - 연두색이나 분홍색 선의 경우에는 logistic regression을 통해 얻을 수 있는 decision boundaries의 예시이다
        + 이는 generalize가 너무 안 된 예시들이다
    - SVM을 통해 얻어낸 검정색 선의 경우에는 더 안전한 decision boundary이다
    - 수학적으로 검은 선은 training examples에 가장 긴 거리(큰 margin)을 갖고있다
    ![SVM decision boundary graph with margin](figs\fig12-12.png)
    - Margin을 가장 크게 만들었기 때문에 더 견고환 분류가 가능해진다
* C가 매우 클 때에 대해 알아봤다
    - 큰 margin을 가진 경우 SVM은 이상치에 대해 매우 민감하게 반응한다
    ![SVM decision boundary graph with outliers](figs\fig12-13.png)
    - 이상치 하나가 classification boundary에 매우 큰 영향을 미칠 위험이 있다
        + 샘플 하나에 의해 알고리즘이 크게 바뀌는 것은 그리 좋지 못하다
        + C값이 매우 크다면 decision boundary가 검정색 선에서 분홍색 선으로 변경될 것이다
        + C값이 충분히 작거나 그리 크지 않다면 검은 선을 유지할 것이다
    - Non-linear하게 분류해야하는 경우에는 어떻게 되는가
        + 적당한 C값을 사용하면 SVM은 잘 작동할 것이다
    - 이것은 몇몇 이상치를 무시하며 작동된다는 의미이다

---

## Large margin classification mathematics(optional)

### SVM decision boundary

![Alt text](figs\fig12-10.png)
* 두개의 단순화를 통한 예시
    - $\theta_0 = 0$
    - n=2
* 2개의 parameter만을 가지고 있기 때문에 cost function은 다음과 같다
$\\\displaystyle \frac{1}{2}(\theta_1^2 + \theta_2^2)$
* 이를 조금 변형시키면 다음과 같다
$\\\displaystyle \frac{1}{2} \left(\sqrt{\theta_1^2 + \theta_2^2} \right)^2$
* 위의 식에서 $\sqrt{\theta_1^2 + \theta_2^2} = ||\theta||$ 이므로 다음과 같이 나타낼 수 있다
$\\\displaystyle \frac{1}{2}||\theta||^2$
* SVM minimizing은 squared norm이다
* $\theta^Tx$ 는 무엇을 의미하는가
    - $\theta^Tx = \theta \cdot x$ 를 의미하므로 다음과 같이 쓸 수 있다
    $\\\theta_1x_1^{(i)} + \theta_2x_2^{(i)} = p^i \times ||\theta||$
* 이를 정리하면 다음과 같다</br>
![SVM cost function with dots products](figs\fig12-14.png)

* 실제 동작</br>
![How SVM select decision boundary](figs\fig12-15.png)
* Decision boundary와 수직인 벡터 $\theta$ 와 각 training data와의 내적을 구한다
* 이 값이 최대가 되는 Decision boundary를 구한다
    - $x^{(1)}$ 의 경우 $p^{(1)}$ 의 크기가 매우 작다
        + 이런 경우 $\theta$ 의 크기가 매우 커지게 된다($x^{(1)} \cdots ||\theta||\geq 1$ 이어야 되기 때문)
    - $x^{(2)}$ 의 경우 $p^{(2)}$ 의 크기가 매우 작다
        + 이런 경우 $\theta$ 의 크기가 매우 커지게 된다($x^{(1)} \cdots ||\theta||< -1$ 이어야 되기 때문)
    - 때문에 SVM은 이 decision boundary를 선택하지 않는다

---

## Kernels - 1: Adapting SVM to non-linear classifiers
* Kernal이 뭐고 어떻게 사용하는가
    - Training set이 있고, non-linear boundary를 찾고싶다</br>
    ![Non-linear Training set](figs\fig12-16.png)
    - Data를 fitting하기 위해 복잡한 polynormial features set을 만든다
        + $h_\theta(x)$ 에 대하여
            + parameter vector와의 연산 결과가 0 이하이면 1을 반환하고 아니면 0을 반환한다
        + 이를 다르게 표현할 수 있는 방법
            + 새로운 feature vector f와 parameter vector를 곱한 값의 합을 통해 $h_\theta(x)$ 를 구할 수 있다
                - $h_\theta(x) = \theta_0 + \theta_1 f_1 + \theta_2 f_2 + \dots$
                    * $f_1 = x_1$
                    * $f_2 = x_1 x_2$
                    * $f_3 = \dots$
                - 그렇게 특별한 값이 필요하지는 않지만 각각의 항은 복잡한 polynomial function이다
        + 고차항인 f를 잘 선택할 수 있는 방법이 있는가
            + 최고차항의 계수가 커질수록 계산에 필요한 비용이 증가한다
* New features
    - 3개의 features를 가진 예씨
    - $x_1$ 에 대한 $x_2$ 의 그래프이다
    - Landmarks라고 하는 3개의 점을 선택한다
        + $l^1, l^2, l^3$
    - $\displaystyle f_1 = \exp \left(-\frac{||x-l^{(1)}||^2}{2\sigma^2} \right)$
        + $||x-l^{(1)}||$ : x와 $l^1$ 사이의 euclidean distance
            + $\displaystyle ||x-l^{(1)}||^2 = \sum_{j=1}^n (x_j-l_j^{(1)})^2$
                - $x$ 와 $l^{(1)}$ 사이의 거리의 제곱
        + $\sigma$ : 표준편차(standard deviation)
        + $\sigma^2$ : 분산(variance)
    - 이 simularity fucntion을 kernel이라고 한다
        - 이경우에는 Gaussian kernel이라고 한다
    - 이 features는 landmark와 x의 거리를 의미한다

### What does $\sigma$ do?

* $\sigma^2$ 은 Gaussian kernel의 parameter이다
    - landmark 주변의 완만한 정도를 나타낸다
* 아래의 그래프는 각각 $\sigma^2 = 1, \sigma^2 = 0.5, \sigma^2 = 3$ 을 나타낸 것이다
![Gaussian Kernel with different sigmas](figs\fig12-17.png)
* $\sigma^2$ 의 값이 작을 수록 더 급격하게 0이 된다
* 예시
    - $\theta_0 + \theta_1 f_1 + \theta_2 f_2 + \theta_3 f_3 \geq 0$
        + $\theta_0 = -0.5$
        + $\theta_1 = 1$
        + $\theta_2 = 0$
        + $\theta_3 = 0$
    - 분홍색 점에 대해서</br>
    ![Gaussian Kernel with a data](figs\fig12-18.png)
        + $f_1$ 은 1에 가까운 값일 것이고, $f_2, f_3$ 는 0에 가까운 값일 것이다
            + $\theta_0 + \theta_1 f_1 + \theta_2 f_2 + \theta_3 f_3$
            + $= -0.5 + 1 + 0 + 0 = 0.5$
                - 0.5는 0보다 크기 때문에 해당 data는 1로 예측할 것이다
    - 하늘색 점에 대해서</br>
    ![Gaussian Kernel with another data](figs\fig12-19.png)
        + 이 경우에는 $-0.5 + 0 + 0 + 0 = -0.5 < 0$ 이므로 0으로 예측할 것이다
    - $l^1, l^2$ 와 가까운 점은 1로 예측하고 $l^3$ 과 가까운 점은 0으로 예측하도록 parameter가 설정되면 아래와 같은 non-linear decision boundary를 얻을 수 있을 것이다</br>
    ![Gaussian Kernel with decision boundary](figs\fig12-20.png)
        + Boundary 내부 : y=1로 예측
        + Boundary 외부 : y=0으로 예측
* SVM에서 어떻게 landmark를 이용해 non-linear boundary를 만들고 kernel 함수를 사용하는지에 대해 알아봤따
* 하지만 이를 적용하기 위해 필요한 것들이 있다
    - 어떻게 landmarks를 정할 것인지
    - Gaussian kernel이 아닌 다른 kernel을 사용할 것인지

---

## Kernels II

* Kernels에 대한 detail에 대해 알아볼 것이다
    - Landmarks를 선택하는 것
        + Landmark를 어디에서 구하는가
    - Kernel을 정의하는 것
    - hypothesis function을 만드는 것

### Choosing the landmarks

* Training data를 선택한다
* 각각의 data에 대해서 동일한 landmark를 설정한다
* m개의 landmarks를 얻을 수 있다
    - Training examples 각각의 위치에 각각 하나의 landmark
    - 이는 featrues가 training set example과 얼마나 떨어져있는지 측정한다는 뜻이다
* 모든 f를 계산한다
    - $f_0$ 부터 $f_m$ 까지의 feature vector f
        + $f_0$ 는 항상 1이다
* f vector를 계산한다
    - $(x^i, y^i)$ 를 이용하여 각각을 계산할 수 있다
        + $f_1^i = k(x^i, l^1)$
        + $f_2^i = k(x^i, l^2)$
        + $\dots$
        + $f_m^i = k(x^i, l^m)$
        + i==m 인 경우에는 1이 된다
    - f vector는 $(f_0, f_1, \dots, f_m)$ 이므로 $[m+1 \times 1]$ 차원의 크기를 갖게 된다

### SVM hypothesis prediction with kernels

* $\theta^Tf \geq 0$ 이면 y=1로 예측한다
    - 어떻게 $\theta$ 를 계산하는가

### SVM training with kernels

![SVM cost function with f](figs\fig12-21.png)
    - x대신 f를 사용하여 minimize를 수행한다
    - 위의 식을 풀면 SVM의 parameter를 계산할 수 있다
* 다른 알고리즘으로도 kernel을 해결할 수 있지만 SVM이 더 효율적이다

#### SVM parameters(C)
* Bias와 variance는 traid off관계이다
* C를 선택해야한다
    - C가 크면
        + low bias, high variance : overfitting
    - C가 작으면
        + high bias, low variance : underfitting

#### SVM parameters($\sigma$)
* f를 계산하기위한 parameter
    - $\sigma^2$ 이 크면
        + f가 smooth해진다
            + higher bias, lower variance
    - $\sigma^2$ 이 크면
        + f가 들쭉날쭉해진다
            + lower bias, higher variance

---

## SVM - implementation and use

### Choosing a kernel

* Kernel의 종류는 다양하다

* Gaussian Kernel
    - 주로 n이 작고 m이 클때(큰 2차원 training set) 선택한다
    - feature들의 값의 크기가 많이 차이 난다면 전처리를 통해 값을 비슷하게 만들어준다
* Linear Kernel
    - Kernel을 사용할 수 없을 때 사용한다
        + 따라서 f 벡터가 없다
    - 주로 n이 크고 m이 작을 때 사용한다
* Polynomial Kernel
* String Kernel
    - 입력이 test string일 대 사용한다
    - Text classification에서 사용한다
* Chi-squared Kernel
* Histogram intersection Kernel

### Logistic regression vs. SVM

* Logistic regression과 SVM중 어떤 것을 사용하면 좋을지는 n과 m을 통해 결정한다
* n이 m보다 크다면
    - Logistic regression이나 SVM
* n이 작고 m이 중간이면
    - Gaussian kernel이 좋다
* n이 작고 m이 크다면
    - Gaussian kernel은 속도가 느리므로 Linear kernel SVM과 logistic regression을 사용한다