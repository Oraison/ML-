# Logistic Regression

## Classification
* 여기서의 y는 불연속적인 값이다
    - logistic regression algorithm을 개발하기 위해서는 입력이 어떤 형태인지 결정해야한다
* Classification problems
    - Email : spam/not spam
    - Online transactions : 사기인가 아닌가
    - 종양 : 악성인가 아닌가
* Variable in these problems is Y
    - Y는 0또는 1로 표현할 수 있다
        + 0 : 아니다
        + 1 : 맞다
* Binery class problem
    - 추후에 multiclass classification problem를 다룰 것이다
    - 이는 binety classification problem을 확장한 것이다
* 어떻게 classification algorithm을 만들까
    - 종양의 크기에 대한 악성 여부
    - Linear regression을 사용할 수 있다
        + classifier output의 분류점을 정한다
            + 해당 값 이상이면 yes 아니면 no
        + 아래의 예시에서는 정상적으로 작동하는 것으로 보인다
        ![Alt text](figs\fig6-1.png)
* 위의 그래프를 통해 두개의 계층을 하나로 만드는 것이 합리적이라는 것을 알 수 있다
    - 하지만 어떤 종양은 크기가 매우 작은데도 악성일 수 있다
    - 이 함수로는 모든 종양을 분류할 수 없다
* Linear regression에는 또 다른 문제가 있다
    - Y는 0 또는 1이다
    - 하지만 위의 함수는 1보다 큰 값이나 0보다 작은 값을 출력할 수도 있다
* 그러므로 logistic regression은 항상 0이나 1을 출력해야한다
    - Logistic regression은 Classification algorithm이다

---

## Hypothesis representation
* 분류를 위해 필요한 함수는 무엇인가
* 해당 함수는 0에서 1사이의 값을 출력해야한다
    - linear regression에서는 $h_\theta(x) = (\theta^Tx)$ 를 사용했다
    - Classification의 경우에는 $h_\theta(x) = g((\theta^Tx))$ 를 사용한다
        + $g(z)$ 는 어떻게 정의하는가
            + z는 실수이다
        + $\displaystyle g(z) = \frac{1}{1+e^{z}}$
            + 이것은 sigmoid function이거나 logistic function이다
        + $g(z)$ 를 $h_\theta(x)$에 적용하면 $\displaystyle h_\theta(x) = \frac{1}{1+e^{-\theta^Tx}}$ 가 된다
* 0.5를 기점으로 크면 yes, 작으면 no가 된다

### Interpreting hypothesis output
* $h_\theta(x)$ 의 출력은 y=1이 될 확률을 의미한다
    - 예시
        + vector X가 $x_0 = 1$ 이고, $x_1 = $ 종양의 크기일 때
        + $h_\theta(x) = 0.7$ 이라면
            + 해당 환자의 종양은 70%확률로 악성일 것이다
    - 이는 아래와 같이 표현할 수 있다
        + $P(y=1|x;\,\theta)$
    - 이 식이 의미하는 바
        + $\theta$ 에 의해 parameterize된 x가 주어졌을 때 y=1이 나올 확률
            + $\theta^Tx$ 가 주어졌을 때 y가 1이 나올 확률
* 이 문제는 binery classification 이므로 y는 0 또는 1이 된다
    - 그러므로 아래와 같이 표현할 수 있다
        + $P(y=1|x;\,\theta) + P(y=0|x;\,\theta) = 1$
        + $P(y=0|x;\,\theta) = 1 - P(y=1|x;\,\theta)$

---

## Decision boundary
* 함수의 모양을 잘 분석하면 해당함수가 어떻게 연산을 수행하는지 더 잘 알 수 있다
    - sigmoid function을 활용하는 방법
        + 출력이 0.5보다 크다면 y=1이라고 예측할 수 있다
        + 마찬가지로 y=0인 경우를 예측할 수 있다
    - $h_\theta(x)$ 가 0.5보다 큰 경우는 언제인가
        - $g(z)$ 가 0.5 이상일 때는 z가 0 이상일 경우이다
            + 즉, z가 양수라면 g(z)가 0.5 이상이다
                + $z = (\theta^Tx)$ 이므로
                + $\theta^Tx \geq 0$ 면 $h_\theta(x) \geq 0.5$ 이다
* 위의 함수가 y=1이기 위해서는  $\theta^Tx \geq 0$ 이라는 사실을 확인했다
    - 그렇다면 $\theta^T \leq 0$ 이라면 y=0이라고 추정할 수 있다

### Decision boundary
* $h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_2 x_2)$
* 예시
    - $\theta_0$ = -3
    - $\theta_1$ = 1
    - $\theta_2$ = 1
* $\theta$ Matrix는 column vector이다
    - $\theta^T = [-3,1,1]$ 이된다
* 이를 풀어 쓰면
    - z = $\theta^Tx$
    - y=1이라고 가정하면
        + $-3x_0 + 1x_1 + 1x_2 \geq 0$
        + $-3 + x_1 + x2 \geq 0$
* 위의 식을 정리하면
    - if($x_1 + x_2 \geq 3$) then y=1이라고 예측할 수 있다
    - $x_1 + x_2 = 3$ 를 그래프에 경계선으로 나타내면 아래와 같다
    ![Alt text](figs\fig6-2.png)
* 이를 통해 그래프에는 2개의 영역이 존재한다
    - 파란 동그라미 : false
    - 빨간 X : true
    - Line : decision boundary
        + 이 직선은 $h_\theta(x) = 0.5$ 가 되는 점들의 집합이다
            + $-3 + x_1 + x2 = 0$ 일 때, 즉 $x_1 + x_2 = 3$ 일 때</br>
              $\displaystyle g(0) = \frac{1}{1+e^{0}} = \frac{1}{1+1} = \frac{1}{2} = 0.5$ 가 된다
    - Decision boundary는 이 함수의 특성이다
        + 변수와 parameter를 사용해 어떤 data가 없어도 boundary를 만들 수 있다
            + 나중에 parameter값을 결정하기 위해 data를 사용한다
        + $x_1 > 5$ 라면 y=1이다

## Non-linear decision boundaries
* 조금 더 복잡한 non-linear data set에서 logistic regression을 만들기
    - polynomial regress처럼 고차식이 추가된다
        + $h_\theta(x) = g(\theta_0 + \theta_1 x_1 + \theta_3 x_1^2 + \theta_4 x_2^2 )$
        + $\theta^T = [-1,0,0,1,1]$
        + y=1이라면
            + $-1 + x_1^2 + x_2^2 \geq 0$
            + $x_1^2 + x_2^2 \geq 1$
        + decision boundary는 $x_1^2 + x_2^2 = 1$ 이다
            + 0을 중점으로 하는 반지름의 길이가 1인 원이 된다.
* 복잡한 decision parameter를 간단하게 만들어서 복잡한 decision boundary를 만들 수 있다

---

## Cost function for logistic regression
* Fit $\theta$
* Cost Function을 사용하기 위한 optimization object를 정의하고 parameters를 찾는다
    - m개의 training example의 training set
        + 각각의 example은 n+1의 column vector이다
![Alt text](figs\fig6-3.png)
* 아래와 같이 진행된다
    - m개의 training example set
    - 각각의 example은 n+1차원의 feature vector이다
    - $x_0$ 는 1이다
    - $y \in \{0, 1\}$
    - 위의 함수는 parameter $\theta$ 를 기반으로 한다
        + training set으로 어떻게 $\theta$ 를 정하는가
* Linear regression 에서는 $\theta$ 에 대한 함수를 아래와 같이 정의했다
$
\\
\displaystyle
J(\theta) = \frac{1}{m}\sum_{i = 1}^m \frac{1}{2} (h_\theta(x^{(i)}) - y^{(i)})^2
$
* 제곱꼴의 error term 대신 cost()를 정의해서 작성할 수 있다
    - $\displaystyle cost(h_\theta(x),y) = \frac{1}{2}(h_\theta(x)-y)^2$
    - 이를 통해 Linear regression을 풀어낸 것 처럼 cost를 구할 수 있다
    * $J(\theta)$ 를 다시 정의하면
    $
    \\
    \displaystyle
    J(\theta) = \frac{1}{m}\sum_{i = 1}^m cost(h_\theta(x^{(i)}),y^{(i)})
    $
        + 이것은 각각의 training data의 cost값을 모두 더한 값이다
        + 이 식이 의미하는 바는 실제 y값과 cost 값의 차이의 평균값이다
        + 이 함수는 parameter obtimization에 대한 non-convex function이다
* Non-confex function이란
    - $J(\theta)$ 라는 함수는 parameters의 값을 결정한다
    - $h_\theta(x)$ 는 non-linearity하다(sigmoid function)
        + 많은 local minimum이 발생할 수 있다
    - convex funcntion이어야 gradient descent를 사용해서 global minimum으로 수렴한다
### A convex logistic regression cost function
* 이를 gradient descent를 적용해 풀기 위해서는 convex cost() function이 필요하다
$
Cost(h_\theta(x),y) = \begin{cases}
-\log(h_\theta(x)) & \text{if\; y = 1}\\
-\log(1-h_\theta(x)) & \text{if\; y = 0}
\end{cases}
$
</br>

$\\h_\theta(x) = P(y=0|x;\, \theta)$ 이고   
$P(y=0|x;\, \theta) + P(y=1|x;\, \theta) = 1$ 이므로    
y=1일 때 $-\log(h_\theta(x))$ 라면   
y=0일 때는 $-\log(1 - h_\theta(x))$ 이다  
* $-\log$   
    - $-$ : cost fucntion 이므로</br>
    - log : 0~1까지의 범위만 다루기 위해서

* Cost function의 결과는 penalty이다
* y=1일 때
    - cost는 $-\log(h_\theta(x))$ 이다
    - $h_\theta$ 가 1일 때 cost가 0이 된다
        - $h_\theta$ 가 1에서 멀어지면(0에서 가까워지면) 즉, 더 틀릴 수록 ocst가 증가한다
            + y=1, $h_\theta = 1$ 이면 coresponds는 0이다
            + $h_\theta$ 가 0으로 가까워지면
                - Cost가 무한히 커진다

---

## Simplified cost function and gradient descent
* logistic regression을 dost function과 gradient descent를 활용해 푸는 법
    - 최종적으로  logistic regression function을 완벽히 구현해야한다
* Logistic regression cost function은 아래와 같다
$
\\\displaystyle
J(\theta) = \frac{1}{m}\sum_{i=1}^mCost(h_\theta(x^{(i)}),y^{(i)}) \\
Cost(h_\theta(x),y) = \begin{cases}
-\log(h_\theta(x)) & \text{if\; y = 1}\\
-\log(1-h_\theta(x)) & \text{if\; y = 0}
\end{cases}
$
* 이것은 하나의 예시이다
    - Binery Classification Problemns의 y는 항상 0 또는 1이다
        + 이런 특성 때문에 cost function을 더 간결하게 만들 수 있다
            + 위의 식은 2개의 case에 대한 2개의 식이다
            + 이를 1개의 식으로 나타내면 더 효율적이다
    - $cost(h_\theta(x), y) = -y\log(h_\theta(x)) - (1-y)log(1-h_\theta(x))$
        + y=0이면 앞의 식이 사라지고(0이 되고) y=1이면 뒤의 식이 사라진다
* 이를 통해 식을 다시 세운다면 아래와 같다
$
\\\displaystyle
J(\theta) = -\frac{1}{m}\sum_{i=1}^m[y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})log(1-h_\theta(x^{(i)}))]
$
* 왜 다른 cost function말고 이 식을 선택했는가
    - 이 cost function은 maximum likelihood estimation의 원리를 이용한 통계에서 도출할 수 있다
        + maximum likelihood estimation
            + 확률변수에서의 값을 토대로 가능도를 최대로 하는 parameter를 구하는 방법  
                log함수는 증가하거나 감소하거나 하나만 하는 함수이므로  
                원래 증가했다면 log도 증가할 것이고  
                원래 감소했다면 log도 감소할 것이므로  
                log를 사용하였을 때의 최대와 log를 사용하지 않았을 때의 최대가 같다는 것을 가정하고 푸는 기법
        + 이는 features의 분포에 Gaussian asssumption이 있다는 것을 의미한다
        + 위의 함수는 convex하다
* To fit parameters $\theta$:
    - $J(\theta)$ 를 최소화 하는 $\theta$ 를 찾는다
* 이를 통해 얻은 결과는 $\theta$ 로 parameterize된 x가 주어졌을 때 y가 1이 될 확률이다

### How to minimize the logistic regression cost function
* Gradient descent를 이용하여 Linear regression을 풀었던 것과 동일하게 푼다
    - 각각의 $\theta_j$ 값을 learning rate를 이용하여 동시에 update한다
    - $h_\theta(x)$ 의 형태가 다르다는 것을 제외하면 둘을 동일하게 해결할 수 있다

---

## Advanced optimization
* 조금 더 나아가서 cost function for logistic regression의 최소화를 할 것이다
    - 큰 규모의 hacine learning에서 좋다(e.g. huge feature set)
* gradient descent의 실제 동작
    - $J(\theta)$ 라는 cost function이 있고, 이를 최소화해야한다
    - 입력으로 $\theta$ 가 들어오면 아래의 과정을 수행하는 코드를 만든다
        1. $J(\theta)$
        2. 모든 j에 대해서($\theta_j$(j=0 to n)) $J(\theta)$ 를 편미분한다
    - 위의 두가지 동작을 하는 코드가 주어진다면
        + $\theta_j$ 를 반복해서 gradient descent로 update한다
    - 각각의 $\theta_j$ 를 순차적으로 update한다
    - 꼭 해야하는 일은 아래와 같다
        + $J(\theta)$ 와 이에대한 미분을 계산하는 코드가 있어야 한다
        + 이를 gradient descent에 적용한다
* cost function minimization을 수행할 때 gradient descent 대신 아래의 방식으로도 해결할 수 있다
    - Conjugate gradient
    - BFGS (Broyden-Fletcher-Goldfarb-Shanno)
    - L-BFGS (Limited memory - BFGS)
* 위의 방식은 같은 입력을 통해 cost function을 minimize하는 조금 더 효율적인 알고리즘이다
* 이것은 매우 복잡한 알고리즘이다
* 장점
    - Learning rate를 지정할 필요 없다
        + 많은 alpha valus를 시도하고 그중 좋은 것 하나를 선택하는 내부 loop을 가지고있다
    - 대게 gradient descent보다 빠르다
        + 단순하게 learning rate를 출력하는 것 이상을 한다
    - 그 복잡한 것을 이해하지 않아도 잘 쓸 수 있다
* 단점
    - 디버깅이 더 힘들다
    - 저것 만으로는 구현하기 힘들다
    - 다른 라이브러리를 사용하면 구현이 달라진다

---

## Multiclass classification problems
* logistic regression for multiclass classification은 모 아니면 도를 사용한다
* Multiclass : yes, no(1, 0) 이상의 것이 필요하다
    - 분류될 class가 여러개이다
* 예시
    - 세모, 네모, 가위표를 분류한다
    - $h_\theta^1(x)$ 는 1이면 세모, 0이면 네모이거나 가위표이다
        + $P(y=1 | x_1; \theta)$
    - $h_\theta^2(x)$ 는 1이면 네모, 0이면 세모이거나 가위표이다
        + $P(y=1 | x_2; \theta)$
    - $h_\theta^3(x)$ 는 1이면 가위표, 0이면 네모이거나 세모이다
        + $P(y=1 | x_3; \theta)$
![Alt text](figs\fig6-4.png)


