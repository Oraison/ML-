# Advice for applying Machine Learning

* 여러 방법을 배웠다
    - 알고있는 것과 어떻게 구현하는지 아는 것은 다른 것이다
    - 각각의 상황에 따라서 어떤 방법으로 구현할지를 결정하는 방법을 숙지해야한다

### Debugging a learning algorithm

* Regularized linear regression을 통해 집 가격을 예측하는 것을 이미 구현했다
    - 이를 실행하고 새로운 data로 test를 해보면 너무 큰 error가 발생한다
* 이후에 어떻게 해야하는가
    - Training data를 더 수집한다
    - Features의 set을 작게 한다
    - Features를 추가한다
    - 고유의 새롭고 더 나은 Featrues를 만든다
    - $\lambda$ 를 키우거나 줄인다
* Machine learning diagnostics

---

## Evaluating a hypothesis

* Parameters를 training data에 fitting해서 error를 최소화한다
    - Error가 작은게 항상 좋다고 생각 할 수 있지만 그렇지 않다
        + Overfitting 문제가 존재한다
    - Hypothesis가 overfitting이라는 것을 어떻게 알 수 있는가
        + 그래프를 그려보면 된다
            + Features의 수가 너무 많으면 그래프로 나타내기 힘들다
* Hypothesis를 평가하는 방법은 다음과 같다
    - Data를 2개로 나눈다
        + 첫번째는 training set이다
        + 두번째는 test set이다
    - 일반적으로 training:test의 비율은 7:3이다
* Data가 정렬되어있으면 random percentage를 보낸다
    - 아니면 무작위로 정렬하고 선택하는 방법도 있다
* 대다수의 train과 test는 다음과 같다
    1. Training data를 이용해 $\theta$ 를 학습한다
        + 70%의 training data로 $J(\theta)$ 를 최소화한다
    2. Test error를 계산한다
        + $\displaystyle J_{test}(\theta) = \frac{1}{2m_{test}} \sum_{i=1}^{m_{test}} (h_\theta(x_{test}^{(i)}) - y_{test}^{(i)})^2$
* Logistic regression의 경우
    - 동일하게 70%는 learning에 사용되고 30%는 test에 사용된다
    ![Alt text](figs\fig10-1.png)

---

## Model selection and training validation test sets
* Polynomial의 차수나 regularization parameter를 어떻게 선택할 것인가(Model selection problems)
* Model selection problem
    - Data fitting에 필요한 polynomial의 차수를 선택한다
    ![Alt text](figs\fig10-2.png)
    - d = 선택하고자 하는 최고차항의 차수
        + d=1 : linear
        + d=2 : quardratic
        + $\dots$
        + d=10
        + Model을 선택하고 fitting하고 hypothesis를 얼마나 잘 generalize할 수 있는지 평가한다
    - 그렇게 하면
        + 1번 model은 parameter vector $\theta^1$ 을 만들 것이다(d=1일 때)
        + 2번 model도 동일한 방법으로 다른 $\theta^2$ 를 얻을 수 있다(d=2일 때)
        + 각각의 model에 대해서 동일하게 시행한다
        + 각각의 model에서 얻은 parameter vector를 test set을 통해 error를 측정한다
            + $J_{test}(\theta^1)$
            + $J_{test}(\theta^2)$
            + $\dots$
            + $J_{test}(\theta^{10})$
        + 가장 error값이 작은 model을 확인한다
    - 만약 d=5일 때 error가 가장 작다면
        + d-5 model을 선택하고 generalize를 수행한다
            + $J_{test}(\theta^5)$ 를 사용할 것이다
                - test set을 가장 잘 표현한다 $\neq$ 모든 data를 잘 표현한다
                    * test set에 overfitting 될 수 있다
            + Test set만으로 평가하는 것은 generalize가 잘 되었는지 판단하는 지표가 되기 어렵다
* Improved model selection
    - Training set을 총 3개로 나눈다
        1. Training set(60%) : m
        2. Cross validation(CV) set(20%) : $m_{cv}$
        3. Test set(20%) : $m_{test}$
    - 이전과 마찬가지로 Training error, Cross validation error, Test error를 계산하면 된다
* 오늘날의 학습 기반의 Machine learning
    - 많은 사람들이 test set을 이용하여 model을 선택하고, test error를 사용해 generalization이 되었는지 확인한다
        + Bias analysis를 도출하기 때문에 좋지 않다
        + 매우 큰 test set이 있다면 괜찮을 수도 있다
    - validation set과 training set을 분리하는 것이 더 좋다

---

## Diagnosis - bias vs variance

* 좋지 않은 결과가 나왔다면 대부분 아래의 경우중 하나일 것이다
    - High bias : under fitting problem
    - High variance : over fitting problem
* Model의 차수가 증가할수록 overfitting해진다</br>
![Alt text](figs\fig10-3.png)
* Training error와 Cross validation error 둘 모두 작은 d를 선택한다
    - Cv error가 크다면 d의 값을 더 크거나 더 작게 바꾼다
    - cv error의 변곡점의 d값이 2라면 d=2 model이 더 적합하다
    - d가 너무 작다면 : high bias problem
    - d가 너무 크다면 : high variance problem
* High bias case
    - Training error와 cv error가 모두 크다
    - Training data를 잘 fitting하지 못했다
* High variance case
    - Training error는 작지만 cv error는 크다
        + Overfitting 되었다
    - training set이 잘 fitting되었다

--- 

## Regularization and bias/variance

* Regularization이 bias와 variance에 미치는 영향
![Alt text](figs\fig10-4.png)
* 위의 식은 Regularization을 사용하여 고차식을 fitting하는 방정식이다
    - Parameter의 크기를 줄이는데 사용
    - $\lambda$ 와 관련해서 다음과 같은 3가지 경우가 있을 수 있다
        + $\lambda$ 가 클 때
            + $\theta$ 값이 큰 패널티를 갖게 된다
            + 모든 parameter가 0에 가깝게 된다
            + Hypothesis가 0에 가까워진다
            + high bias -> underfitting data
        + $\lambda$ 가 적당한 값일 때
            + 이 경우에만 의미있는 결과를 도출할 수 있다
        + $\lambda$ 가 작을 때
            + $\lambda = 0$ 
                - Regularization term을 없앤다
            + High variance -> Get overfitting
* **좋은 $\lambda$ 값을 자동으로 선택하는 방법**
    - 2배로 값을 증가시키기도 한다
        + model(1) : $\lambda = 0$
        + model(2) : $\lambda = 0.01$
        + model(3) : $\lambda = 0.02$
        + model(4) : $\lambda = 0.04$
        + model(5) : $\lambda = 0.08$ </br>
        $\vdots$
        + model(p) : $\lambda = 10$
    - 각각의 $\lambda$ model에 번호를 매긴다
    - 각 p에 대하여
        + model($p^{th}$)를 선택한다
        + Cost function을 최소화한다
        + 이를통해 parameter vector $\theta^{(p)}$ 를 얻을 수 있다
            + 각각의 다른 $\lambda$ 값을 통해 연산된 $\theta$ vector set을 얻을 수 있다
        + 모든 hypothesis를 cross validation을 사용해서 확인한다
            + Cross validation set의 error를 구한다
            + 가장 error가 작은 model을 선택한다
                - $\theta^{(5)}$ 가 가장 작다고 가정하자
    - 최종적으로 $\theta^{(5)}$ 를 선택했고 이를 통해 test set을 test하면 된다
* **Bias/variance as a function of $\lambda$**
    - $\lambda : J_{train}$ 
        + $\lambda$ 가 작을 때
            + regularization이 0에 가까워진다
        + $\lambda$ 가 클 때
            + high bias를 갖게 된다
    - $\lambda : J_{cv}$ 
        + $\lambda$ 가 작을 때
            + high variance를 갖게 된다
                - 너무 작은 경우에는 overfitting이 발생한다
        + $\lambda$ 가 클 때
            + Underfitting이 발생한다
                - high bias
            + Cross validation의 error가 증가한다

---

## Learning curves

* Learning curve는 algorithmic sanity checking이나 성능 향상에 유용하다
* Learning curve가 무엇인가
    - $J_{train}$ 이나 $J_{cv}$ 의 그래프
    - Training example의 수 m에 대한 그래프이다
        + m은 상수이다
        + 그렇기 때문에 m의 수를 임의로 줄여 적은 수의 training set의 error를 연산한다
    - $J_{train}$
        + sample의 수가 적을수록 error의 수가 적다
        + m이 증가할수록 error도 증가한다
    - $J_{cv}$
        + Cross validation set의 error
        + Training set의 크기가 아주 작다면 generalize가 잘 되지 않는다
        + Training set의 크기가 증가하면 hypothesis가 더 잘 generalize된다
        + m이 증가하면 cv error는 감소한다
        ![Alt text](figs\fig10-5.png)
* 위의 그래프를 해석한다면
    - High bias(직선)
        + $J_{train}$
            + m이 증가함에 따라 cross validation과 training의 error가 비슷해진다
        + $J_{cv}$
            + 직선은 많은 데이터 중 일부의 데이터에만 적용된다
        + High bias인 경우에는 training error와 cross validation error 둘 모두 크다
        + Learning algorithm이 high bias한 하다면 더 많은 data는 필요하지 않다
            + High bias를 최대한 피하는 것이 좋다
        + High bias는 data를 modeling하는 것에 있어 큰 문제이다
            + 더 많은 data는 model을 향상시키는데 크게 도움되지 않는다
    - High variance(고차식)
        + $J_{train}$
            + set이 작은 경우 error도 작다
            + Training set의 크기가 증가해도 value는 여전히 작다
                * 하지만 조금씩 증가한다
        + $J_{cv}$
            + 적당한 양의 example이 있어도 error는 크다
                * High variance한 경우에는 generalize가 잘 안 되어있기 때문이다
        + 이는 training set error와 cross validation의 error의 차이가 크다는 것을 의미한다
        + Learning algorithm이 hi variance하다면 더 많은 data가 유용할 수 있다
            + 이미 high variance하다면 더 많은 data로 해결될 수도 있다
    
---

## 요약
* Get more example
    - fix high variance
* Smaller set of features
    - fix high variance
* Adding additional features
    - fix high vias
        + hypothesis가 너무 단순하기때문에 발생하는 문제이므로 feature를 추가해서 해결한다
* Add polynomial terms
    - fix high bias
* Decreasing $\lambda$
    - fix high bias
* Increses $\lambda$
    - fix high variance
* validation set
    - layer의 수, unit의 수, 최고차항 등의 hyper parameter를 결정하기 위한 data set
* test set
    - parameter를 결정하기 위한 data set