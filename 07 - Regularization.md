# Regularization

## The problem of overfitting
* 이전까지의 알고리즘은 대다수의 경우에서 잘 작동할 것이다
    - 하지만 overfitting이 발생할 수 있다
        + 하나의 $\theta$ 값이 너무 커지면 하나의 feature에 너무 큰 의존도를 갖게 된다

### Overfitting with linear regression
* house pricing example을 다시 살펴보자
    - Linear function으로 데이터를 표현하는 것 - 그리 좋은 방법은 아니다
        + 이것은 underfitting이다
            + high bias
                - bias : 예측값과 실측값의 오차
        + bias는 역사적이고 기술적인 것이다
            + data를 직선으로 fitting하려면 선형적이어야 한다는 선입견이 있다
    - Fit a quadratic function
        + 잘 동작한다
    - Fit a 4th order polynomial
        + 4차 함수로 만들어진 곡선은 5개의 데이터를 모두 만족시킨다
            - 이것은 training set을 잘 fitting하는 것으로 보인다
            - 데이터들을 모두 만족하지만, 이것은 그리 좋은 모델은 아니다
        + 이것이 overfitting이다 - high variance하다
            - high variance : 입력된 data에 따라 출력이 변화하는 정도
    - 알고리즘은 high variance하다
        + 고차 함수로 데이터를 fitting할 때, 그 함수가 모든 데이터를 만족하게된다 - High variance
        + 함수가 너무 고차원이다
![Alt text](figs\fig7-1.png)
* 요약하자면 너무 많은 features를 사용하려고 하면 cost function이 0이 될 수 있다

### Overfitting with logistic regression
* logistic regression에서도 같은 문제가 발생할 수 있다
    - sigmid function은 underfit이다
    - 너무 고차함수를 사용하면 overfitting이 발생한다
    ![Alt text](figs\fig7-2.png)

### Addressing overfitting
* 언제 overfitting과 underfitting이 일어나는지는 추후에 살펴볼 것이다
* 이전에 고차함수 그래프를 만들었다
    - hypothesis 그래프를 만드는 것은 하나의 방법이지만 항상 효과적이지는 않다
    - 종종 많은 양의 features를 가진다
        + 이 경우에는 최고차항을 결정하는 것 보다 data를 plot하고 시각화 하여 어떤 feature를 유지하고 어떤 featur를 삭제해야 하는지 결정하는 것이 더 어렵다
    - 만약 적은 양의 data에 많은 features가 있다면
        + overffiting이 문제가 될 수 있다
* 어떻게 해야 해결할 수 있을까
    1. feature의 수를 줄인다
        + 직접 feature중에 남길것을 고른다
        +  Model selection algorithms은 추후에 살펴보겠다(features의 수를 줄이는데 좋다)
        + 하지만 features의 수를 줄이면 잃는 정보가 발생한다
            + 손실되는 정보를 최소화하는 것이 이상적이지만, 어쩔 수 없이 정보의 손실이 발생할 수 밖에 없다
    2. Regularization
        + 모든 features를 유지하면서 parameter $\theta$ 의 규모를 줄인다
        + 많은 features가 y를 예측할 때 각각의 feature는 조금만 기여할 때 잘 작동한다

---

## Cost function optimization for regularization
* 몇몇 $\theta$ parameter를 정말 작게 만들고 penalty를 부과한다
    - 예를들어 아래의 식에서 $\theta_3$ 과 $\theta_4$ 의 경우이다
    $\\\displaystyle
    \min_\theta\frac{1}{2m}\sum_{i=1}^m (h_\theta(x^{(i)})-y^{(i)})^2 + 1000(\theta_3)^2 + 1000(\theta_4)^2
    $
* cost function 뒤의 $\theta_3$ 과 $\theta_4$ 에 penalty를 부여한 것이다
    - 위의 식을 풀고 나면 $\theta_3$ 과 $\theta_4$ 의 값은 0에 가깝게 되어있을 것이다
        + constrants가 매우 크기 때문이다
    - 때문에 기존의 2차함수와 비슷한 형태로 나타날 것이다
    ![Alt text](figs\fig7-3.png)
* 위의 예제에서는 2개의 parameter value에 penalty를 부여했다
    - 더 일반적으로 regrularization은 다음과 같다
* Regularization
    - parameter의 값이 작아지면 hypothesis가 단순해진다(차원이 낮아진다)
        + 항의 일부가 제거된다
    - 단순한 hypthesis는 overfitting을 할 가능성이 줄어든다
* Another example
    - 100개의 feature가 있을 때
    - 다항식이 주어졌을 때와는 달리 어떤 항이 최고차항인지 알 수 없다
        + 어떻게 감소시킬 하나의 항을 찾을 수 있을까
    - Regularization을 활용하여 모든 parameter를 감소시켜 cost function을 변경한다
        + 기존의 cost function에 항을 추가한다
            + regularization 항은 모든 parameter를 감소시킨다
            + 보통 $\theta_0$ 는 penalty를 부여하지 않는다
                + $\theta_1$ 이상부터 minimization을 수행한다
* 실제로 $\theta_0$를 포함해도 큰 변동은 없다
$
\\\displaystyle
J_\theta(x) = \frac{1}{2m}\left[\sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda \sum_{i=1}^n(\theta_j)^2 \right]
$
* $\theta_j$ 를 업데이트 할 때
    - $\theta_j := \theta_j - \alpha \times [\theta_j 에\ 관련된\ 항들]$
    $
    \\\displaystyle
    \theta_j := \theta_j - \alpha\frac{1}{m}\left[ \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda\theta_j \right] \\
    \theta_j := \theta_j - \alpha\frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \alpha\frac{\lambda}{m}\theta_j  \\
    \theta_j := \theta_j(1-\alpha\frac{\lambda}{m}) - \alpha\frac{1}{m} \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}\\
    $
* $\displaystyle(1-\alpha\frac{\lambda}{m})$
    - 이 항은 보통 1보다 작은 수가 될 것이다
    - 보통 learning rate는 작고, m은 크기 때문에
        + (1 - 작은 값)이 될 것이다
        + 보통 0.99에서 0.95정도의 값을 지닌다
    - 이것은 $\theta_j$ 에 0.99를 곱한 값이 된다는 뜻이다
        + 때문에 $\theta_j$ 의 값이 아주 조금 작아진다

---

## Regularization with the normal equation
* normal equation은 또 다른 linear regression model이다
    - $J(\theta)$ 를 normal equation으로 minimiae하는 것이다
    - Regularization을 수행하기 위해서는 $+ \lambda[n+1 \times n+1]$ 의 항을 추가하면 된다
        + $[n+1 \times n+1]$ 은 n+1 identity matrix이다

### Regularization for logistic regression
* Linear regression과 동일한 형태를 하게된다

