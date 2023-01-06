# Neural Networks - Learning

## Neural network cost function
* NNs - 가장 강력한 learning algorithms중 하나
    - 주어진 trainingset으로 derived parameters을 fitting하기위한 learning algorithm
    - Neural entwork의 cost function을 살펴보자
* 먼저 classification problems의 NNs를 살펴보자
* Set up
    - Training set : {$(x^1,y^1), (x^2,y^2), (x^3,y^3), \dots, (x^n,y^m)$}
    - L : network의 layers의 수
        + 아래의 예시에서는 L=4이다
    - $s_l$ : l번 layer의 numint 의 개수(bias unit은 포함되지 않는다)
    ![Alt text](figs\fig9-1.png)
* 위의 예시의 경우
    - l = 4
    - $s_1$ = 3
    - $s_2$ = 5
    - $s_3$ = 5
    - $s_4$ = 4

### Types of classification problems with NNs
* classification은 2개로 나눌 수 있다
1. Binary classification
    - 1개의 output(0 or 1)
    - Output node의 수가 1개면 된다
        + 값은 실수일 것이다
    - k = 1
        + k는 output layer의 개수이다
    - $s_L$ = k
2. Multi-class classification
    - k개의 다른 classifications
    - k는 3 이상이다
    - 2개라면 binary classification이다
    - $s_L$ = k
    - y는 k차원의 실수 vector이다

### Cost function for neural networks
* (regularized) logistic regression cost function : 
$
\\\displaystyle
J(\theta) = -\frac{1}{m}\left[ \sum_{i=1}^m y^{(i)} \log h_\theta(x^{(i)}) + (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))\right] + \frac{\lambda}{2m}\sum_{j=1}^n (\theta_j)^2
$
* Neural network에서의 cost function의 generalization은 다음과같다
    - 단일 output이 아니라 k개의 output이 필요하다
$
\\\displaystyle
J(\Theta) = -\frac{1}{m}\left[ \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log h_\Theta(x^{(i)})_k + (1-y_k^{(i)}) \log(1-h_\Theta(x^{(i)})_k)\right] + \frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{sl} \sum_{j=1}^{s_l+1} (\Theta_{ji}^{(l)})^2
$
* 위의 cost function의 output은 k차원의 vector이다
    - $h_\Theta(x)$ 는 k 차원의 vector이므로, $h_\Theta(x)_i$ 는 해당 vector의 원소 중 하나이다
* Cost function $j(\Theta)$
    - logic regression에서 처럼 각 항에 $\frac{1}{m}$ 을 곱한다
    - 하지만 위의 cost function에서는 k=1에서 K까지(K : # of output node)의 합이다
        + Summation은 k개의 output units의 합이다
    - 매우 복잡해보이지만 그리 어렵지는 않다
        + Bias term은 계산에 포함되지 않는다
            + Bias term을 함께 계산하더라도 크게 문제되지는 않는다
* 위의 식은 크게 두 항으로 나눌 수 있다

### First half
$
\\\displaystyle
-\frac{1}{m}\left[ \sum_{i=1}^m \sum_{k=1}^K y_k^{(i)} \log h_\Theta(x^{(i)})_k + (1-y_k^{(i)}) \log(1-h_\Theta(x^{(i)})_k)\right]
$
* 위의 식이 의미하는 바
    - 각각의 training data example(1부터 m까지 : 첫 summation)
        + 각 output vector의 opsition의 합
* logistic regression의 합의 평균이다

### Second half
$
\\\displaystyle
\frac{\lambda}{2m} \sum_{l=1}^{L-1} \sum_{i=1}^{sl} \sum_{j=1}^{s_l+1} (\Theta_{ji}^{(l)})^2
$
* 이것은 큰 규모의 regularization summation 항이다.
    - 간단한 triple nested summation이다
* 이를 weight decay 항이라고도 부른다
* lambda value는 두 항의 중요도를 설정하기 위해 사용된다
* 어떻게 하면 이 cost function을 최소화 할 수 있는가

---

## Summary of what's about to go down
* Forward propagation은 이미 알고있다
    - 이 알고리즘은 neural entwork와 initial input을 network에 입력하는 것이다
        + 그 결과 output hypothesis를 생성하는데, 이것은 실수일 수도 있고 vector일 수도 있다
* Back propagation에 대해 설명할 것이다
    - Back propagation은 기본적으로 network를 통해 만들어낸 output과 실제값 y를 비교해서 network가 얼마나 잘못되었는지를 계산한다
        + network가 얼마나 잘못되었는가 = parameter가 얼마나 잘못되었는가
    - 이를 통해 얻은 error를 통해 이전 layer(layer L-1)에서의 오류를 역산한다
    - 이 과정을 input layer까지 반복한다
        + error가 없다면 activation이 input과 동일할 것이다
    - 각 unit에서의 error를 측정하면 partial derivatives를 계산할 수 있다
        + partition derivatives는 gradient descent가 cost function을 최소화 하기위해 필요하기 때문이다
    - Gradient descent의 cost function을 최소화하고 $\Theta$ 를 update하기 위해 partial derivatives를 사용한다
    - 이를 gradient descent가 수렴할 때 까지 반복한다

---

## Back propagation algorithm
* 동작 원리</br>
![Alt text](figs\fig9-2.png)
    1. Layer k-1(output 직전 unit) </br>
![Alt text](figs\fig9-3.png)
        + error에 $w_{10}^{(1)}$ 이 미치는 영향 = error에 $a_{20}$ 이 미치는 영향 $\times$ $a_{20}$ 에 $z_{20}$ 이 미치는 영향 $\times$  $z_{20}$ 에 $w_{10}^{(1)}$ 이 미치는 영향
            + 해당 unit이 영향을 미치는 output이 하나만 존재한다
                - $w_{10}^{(0)} := w_{10}^{(0)} - \alpha \frac{\partial E_{tot}}{\partial w_{10}^{(0)}}$
                    * 해당 연산은 모든 $E_{tot}$ 을 연산한 이후에 동시에 업데이트 한다
    2. 그 외 </br>
![Alt text](figs\fig9-4.png)
        + error에 $w_{10}^{(0)}$ 이 미치는 영향 = ($y_1$의 error에 $a_{10}$ 이 미치는 영향 + $y_2$의 error에 $a_{10}$ 이 미치는 영향) $\times$ $a_{10}$ 에 $z_10$ 이 미치는 영향 $\times$ $z_{10}$ 에 $w_{10}^{(0)}$ 이 미치는 영향
            + 해당 unit이 영향을 미치는 output이 2개 존재한다
                - $E_{tot} = E_1 + E_2$
                - $\displaystyle \frac{\partial E_{tot}}{\partial w_{10}^{(0)}} = \left( \frac{\partial E_{y_1}}{\partial a_{10}} + \frac{\partial E_{y_2}}{\partial a_{10}} \right) \frac{\partial a_{10}}{\partial z_{10}} \frac{\partial z_{10}}{\partial w_{10}^{(0)}}$ 
                    * $E_{y_1}$ 과 $E_{y_2}$ 를 각각 계산해서 더해준값이 $E_{tot}$ 이 된다

### Gradient computation

* Layer 1
    - $a^{(1)} = x$
    - $z^{(2)} = \Theta^{(1)}a^{(1)}$
* Layer 2
    - $a^{(2)} = g(z^{(2)}) (+ a_0^{(2)})$
    - $z^{(3)} = \Theta^{(2)}a^{(2)}$
* Layer 3
    - $a^{(3)} = g(z^{(3)}) (+ a_0^{(3)})$
    - $z^{(4)} = \Theta^{(3)}a^{(3)}$
* Layer 4
    - $a^{(4)} = h_\Theta(x) = g(z^{(4)})$ </br>
![Alt text](figs\fig9-5.png)

#### Sigmoid function derivative
$
\\\displaystyle
\frac{1}{1+e^{-x}} = \frac{1}{1+e^{-x}}\frac{e^x}{e^x} = \frac{e^x}{e^x+1} \\
$

$
\displaystyle
\begin{aligned}
\frac{d}{dx}\sigma(x) =& \frac{d}{dx}\left[ \frac{1}{1+e^{-1}} \right] = \frac{d}{dx}(1+e^{-x})^{-1}\\
=& -1 \times(1+e^{-x})^{-2}(-e^{-x}) \\
=& \frac{-e^{-x}}{-(1+e^{-x})^2} \\
=& \frac{e^{-x}}{(1+e^{-x})^2} \\
=& \frac{1}{1+e^{-x}} \frac{e^{-x}}{1+e^{-x}} \\
=& \frac{1}{1+e^{-x}} \frac{e^{-x}+(1-1)}{1+e^{-x}} \\
=& \frac{1}{1+e^{-x}} \frac{(1+e^{-x})-1}{1+e^{-x}} \\
=& \frac{1}{1+e^{-x}} \left[ \frac{(1+e^{-x})}{1+e^{-x}} - \frac{1}{1+e^{-x}}\right] \\
=& \frac{1}{1+e^{-x}} \left[ 1 - \frac{1}{1+e^{-x}}\right] \\
=& \sigma(x)(1-\sigma(x))
\end{aligned}
$

---

## Random initalization

* 모든 $\Theta$ 값을 random한 작은 초기값으로 설정한다
    - 0으로 시작할 경우(Linear regression에서의 경우) 알고리즘이 실패할 수 있다
        + 각 layers의 모든 activation values가 같아진다
* 그러므로 random한 값을 고른다
    - 0에서 1 사이의 값을 선택하고 $\epsilon$ 으로 scale을 조정한다($\epsilon$ 은 constant이다)

---

## Putting it all together

1. pick a =network architecture
    - 개수
        + Input units
            + x의 수(feature vector의 dimensions)
        + Output units
            + 분류할 classes의 수
        + Hidden units
            + 기본적으로는 1개의 hidden layer를 사용한다
            + 각각의 layer마다 같은 수의 unit을 사용한다
            + input features의 1.5배 또는 2배의 units을 사용할 수 있다
            + 보통 hidden units의 수가 많은 것이 더 좋다
                - hidden units의 수가 많아질 수록 계산 비용이 더 높아진다
2. Training a neural network
    1. Weights를 random하게 초기화한다
        + 0에 가까운 작은 값으로 진행한다
    2. 모든 $x^{(i)}$ 에 대한 $h_\Theta(x)^i$ 를 구하기 위한 forward propagation을 구현한다
    3. Cost function $J(\Theta)$ 를 계산하기 위한 코드를 구현한다
    4. 편미분을 계산할 back propagation을 구현한다
    5. 아래의 알고리즘으로 계산된 partial derivatives와 $J(\Theta)$ 의 gradient의 numerical estimation을 비교하기 위해 gradient checking을 사용한다 </br>
        
        for i = 1 to m{ </br>
            Forward propagation on ($x^i$, $y^i$) : get activation($a$) terms </br>
            Back propagation on ($x^i$, $y^i$) : get delta($\delta$) terms </br>
            Compute $\Delta := \Delta^l + \delta^{l+1}(a^l)^T$
        }
    6. $J(\Theta)$ 를 최소화 하기 위해 gradient descent나 advanced optimization method를 사용해 back propagation을 푼다