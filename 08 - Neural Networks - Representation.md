# Neural Networks - Representation

## Neural networks - Overview and summary

### Why do we need neural networks?

* 복잡한 supervised learning classification problem이 있을 때
    - logistic regression을 polynomial terms로 해결할 수 있을까
    - 1-2개의 feature에서는 할 수 있을 것이다
    - 100개의 feature가 있다면?
* housing example
    - 100개의 house features로 6개월 내에 집이 팔릴지를 예측한다
    - 모든 항이 2차항인 경우
        + $(x_1^2, x_1x_2, x_1x_4, \dots, x_1x_100)$ 같은 항들이 있을 것이다
        + feature의 수에 따라 $O(n^2)$ 의 복잡도를 갖는다
        + 이를 계산하려면 많은 시간이 소요된다
    - features의 일부만 사용하는 방법
        + 이경우 features의 수가 충분하지 않으면 복잡한 datasset에 대해서 표현하기 힘들어진다
    - 모든 항이 3차항인 겨우
        + feature의 수에 따라 $O(n^3)$ 의 복잡도를 갖는다
* n이 클 때 classifiers를 만드는 것은 그리 좋은 방법이 아니다

### Example: Problems where n is large - computer vision
* Computer vision은 픽셀 값의 matrix로 보인다
* 차량 감지기를 만든다고 했으 ㄹ때
    - training set에는 다음의 정보가 있을 것이다
        + 자동차인 부분
        + 자동차가 아닌 부분
    - 그리고 자동차를 반복해서 학습할 것이다
* 어떻게 할 수 있는가
    - 두 필셀을 표시한다(픽셀의 좌표)
    - 자동차인지 아닌지 그래프에 표시한다
* 이를 분류하기 위해서는 non-linear hypothesis가 필요하다
* feature space
    - 만약 $50 \times 50$ 픽셀이 있다면 2,500개의 픽셀이 필요하믈 n=2500이다
    - RGB를 고려한다면 n=7500이다
* n이 너무 크다
    - 단순한 logistic regression은 매우 복잡한 시스템에 적합하지 않다
    - Neural network는 feature space가 매우 크고 복잡한 non-linear hypothesis에 적합하다

### Neurons and the brain
* Neural networks(NNs)는 뇌의 동작을 따라하는 기계에서 시작되었다
* Origins
    - learning system을 만들 때 왜 뇌를 모방하면 안되나
    - 80년대에서 90년대에는 그렇게 했다
    - 90년대 후반에 인기가 떨어졌다
    - 최근에 다시 화두가 되었다
        - NNs는 계산에 오래걸린다
            - 그래서 최근들어서야 큰 규모의 NNs를 계산할 수 있게 되었다
* Brain
    - 뇌가 하나의 learning algorithm을 가지고 있다는 가설에서 출발한다
    - 가설의 증거
        + 청각피질에서 sound signal을 인식할 때
            + 귀에서 청각피질까지의 신경을 절단하면
            + 시신경이 청각피질로 재연결 된다
            + 시신경이 보는 것을 배운다
        + 촉각(touch procession)
            + 시신경을 감각피질로 연결하면 보는 것을 배운다
* 다른 예시
    - Human echolocation
        + 시각 장애인들은 소리와 echo를 구분하는 훈련을 한다
        + 이를 통해 주변을 걸을 수 있다
    - Haptic belt direction sense
* 뇌는 모든 소스에서 데이터를 처리하고 학습할 수 있다

---

## Model representation I
* Neural network를 표현하는 방법
    - NNs는 neurones의 network를 시뮬레이션 하는 방법으로 개발했다
* Neurone은 어떻게 생겼는가
    - 세가지 부분으로 나눌 수 있다
        + cell body
        + input wires(dendrites)
        + output wire(axon)
    - 비슷하게
        + Neurone은 1개 이상의 Dendrites를 통해 input을 받는다
        + processing
        + Axon으로 output을 보낸다
    - Neurone은 전기를 통해 통신한다
        + Axon을 통해 다른 neurone으로 전기 펄스를 전달한다

### Artificial neural network - representation of a neurone
* Artifical Neural Network에서 neurone은 logical unit이다
    - input wires를 통해 입력이 제공된다
    - logical unit이 계산한다
        + logistic 계산 과정은 이전의 logistic regression hypothesis연산과 같다
    - output wires로 출력을 내려보낸다
![Alt text](figs\fig8-1.png)
* 매우 간단한 neuron의 계산이다
    - 종종 $x_0$ input을 포함하는 것이 좋기도 하다 - bias unit
    - 이 artifical neurone은 sigmoid activation function이다
        - $\theta$ vector는 이 모델의 weights라고 부리기도 한다
    - 위의 diagram은 single neurones이다
        - 아래에는 같이 반응하는 neurones의 그룹이 있다
![Alt text](figs\fig8-2.png)
* input은 $x_1, x_2, x_3$ 이다
    - 첫번째 layer를 input activation이라고 부른다($a_1^1, a_2^1, a_3^1$)
    - 두번째 layer에는 3개의 neurones이 있다($a_1^2, a_2^2, a_3^2$)
    - 마지막 4번째 neurone은 출력을 만든다
        + 이를 $a_1^3$라고 부를 수 있다
* 첫번째 layer는 input layer이다
* 마지막 layer는 output layer이다
    - hypothesis로 계산되어 만들어진 value
* Middle layer(s)는 hidden layers이다
    - hidden layers에서 처리된 값들은 확인할 수 없다

### Neural networks -notation
* $a_i^{(j)}$
    - j layer의 i번째 unit의 activation
    - $a_1^2$ 는 2번째 layer의 1번째 유닛의 activation이다
    - Activation은 해당 Node에서 계산돼 나온 value를 의미한다
* $\Theta^{(j)}$
    - layer j에서 layer j+1로의 functioin mapping을 제어하는 parameter의 matrix
    - 한 layer에서 다음 layer로의 mapping을 제어하기 위한 parameter
    - If network has
        + layer j에 $s_j$ 개의 units가 있다
        + layer j+1에 $s_{j+1}$ 개의 units가 있다
        + 이런 경우 $\Theta^j$ 는 [$s_{j+1} \times s_j + 1$] 차원의 matrix가 된다
    - $\Theta$ matrix
        + Column의 길이는 다음 layer의 units의 개수이다
        + row의 길이는 현재 layer의 units의 개수 + 1이다
            + bias unit이 있기 때문
        + 각각의 unit의 수가 101, 21 개인 두 layer가 있다면
            + $\Theta^j$ 는 [$21 \times 102$] 차원의 matrix가 된다
* 어떤 연산이 발생하는가
    - 각각의 node에서 activation을 계산한다
    - Activation은 아래의 요소에 의해 결전된다
        + node의 input(s)
        + node와 연관된 parameter
            + 해당 layer와 연관된 $\Theta$ vector로 부터 얻어진다
* notwork의 예시와 각각의 Node에 대한 계산
![Alt text](figs\fig8-3.png)
    - layer 2의 각각의 activation을 bias term과 input values를 기반으로 계산한다
        + $x_0$ 부터 $x_3$
    - input이 x가 아닌 이전 layer의 activation values라는 것을 제외한다면 동일하게 작동하여 final hypothesis(layer 3의 node)를 계산한다
* 각 hidden units의 activation 값($a_1^2$)은 input linear combination에 적용되는 sigmoid function과 동일하다
    - 3개의 input units
        + $\Theta^{(1)}$ 은 input units에서 hiddne units로의 mapping을 제어하는 parameter matrix이다
            + 위의 예시에서는 [$3 \times 4$] 차원 matrix이다
    - 3개의 hidden units
        + $\Theta^{(2)}$ 는 hiddne units에서 output layer로의 mapping을 제어하는 parameter matrix이다
            + 위의 예시에서는 [$1 \times 4$] 차원 matrix이다
    - 1개의 oputput unit
* 중요한 개념은 다음과 같다
    - 모든 input과 activation은 다음 layer의 모든 node로 간다
        + 각각의 **layer tyransition**은 다음과 같은 중요도를 갖는 parameter matrix를 사용한다
            + 이후의 명명법의 일관성을 위해 j, i, l을 변수로 사용하고 있다
                - 이번 section의 말미에 j를 사용하여 현재 확인하고 있는 layer를 나타낼 것이다
            + $\Theta_{ji}^l$
                - j : 1부터 l+1 layer의 units의 개수 까지의 범위를 지닌다
                - i : 0부터 l layer의 units의 개수 까지의 범위를 지닌다
                - l : the layer you're moving **FROM**
* 예시
    - $\Theta_{13}^1$
        + 1(j) : layer l+1(2)의 node 1로의 mapping
        + 3(i) : layer l(1)의 node 3부터의 mapping
        + 1(l) : layer 1부터의 mapping

---

## Model representation II
어떻게 벡터화된 구현을 통해 효율적으로 계산을 수행하는지에 대해 살펴볼 것이다</br>
왜 NNs는 좋은지, 어떻게 complex non-linear 학습하도록 한는지

* 이전에 본 문제가 아래에 있다
    - 아래의 수식은 hypothesis를 출력하기 위한 연산 과정이다
![Alt text](figs\fig8-3.png)

* 추가적인 항의 정의
    - $z_1^2$ = $\Theta_{10}^1 x_0 + \Theta_{11}^1 x_1 + \Theta_{12}^1 x_2 + \Theta_{13}^1 x_3$
        + $z_1^2$ : 2번 layer의 1번 Node의 input
    - 이것은 $a_1^2 = g(z_1^2)$ 임을 의미한다
    - 윗첨자의 숫자는 layer와 연관되어있다
* 비슷하게 다른 것들도 정의할 수 있다
    - $z_2^2, z_3^2$
    - 이 값은 values의 단순히 linear combination이다
 * 위의 블록에서 재정의 된 것
    - neural network computation은 백터화 했다
    - 그러므로 다음과 같이 정의할 수 있다
        + x를 feature vector x로
        + $z^2$ 를 2번째 layer의 z values의 vector로</br>
![Alt text](figs\fig8-4.png)
* $z^2$ 는 [$3 \times 1$] 차원의 vector이다
* 다음의 2개의 과정을 통해 neural network의 computation을 벡터화 할 수 있다
    - $z^2$ = $\Theta^{(1)}x$
        + $\Theta^{(1)}x$ 은 위에서 정의된 matrix이다
        + x는 feature vector이다
    - $z^2$ = $g(z^{(2)})$
        + $z^2$ 와 $a^2$ 모두 [$3 \times 1$] 차원의 vector 이다
        + $g()$ 는 $z^2$ vector의 각각의 elements에 sigmoid(logistic) function을 적용한다
* input layer에 위의 표기법을 적용하면
    - $a^1$ = x
        + $a^1$ 은 input layer의 activations이다
        + input layer의 activation은 input이다
    - 그러므로 x를 $a^1$ 을 사용하여 재정의 할 수 있다
        + $a^1$ 은 input vector이다
        + $a^2$ 는 $g(z^2)$ 함수에 의해 계산된 values의 vector이다
* $z^2$ vector를 계산한 이후 final hypothessis를 계산하기 위해 $a_0^2$ 를 계산할 필요가 있다
    - $h_\Theta(x) = g(\Theta_{10}^{(2)} a_0^{(2)} + \Theta_{11}^{(2)} a_1^{(2)} + \Theta_{12}^{(2)} a_2^{(2)} + \Theta_{13}^{(2)} a_3^{(2)})$
* 추가적인 bias unit을 처리하기 위해서는 $a_0^2 = 1$ 을 추가해야한다
    - $a_0^2$ 를 추가하면 $a^2$ 는 [$4 \times 1$] 차원의 vector가 된다
* 그러므로
    -  $z^3 = \Theta^2a^2$
        + 이것은 위의 식의 일부 항이다
    - $h_\Theta(x) = a^3 = g(z^3)$
* 이러한 과정을 forward propagation이다로 부른다
    - input unit의 activations로 부터 시작한다
    - 각각의 layer의 activation을 순차적으로 계산하고 propagate한다
    - 이 구현의 벡터화 된 버전이다

### Neural networks learning its own features
* 아래의 diagram은 logistic regressioon과 매우 유사하다
![Alt text](figs\fig8-5.png)
* Layer 3은 logistic regression node이다
    - $h_\Theta()$ 의 출력은 $g(\Theta_{10}^{(2)} a_0^{(2)} + \Theta_{11}^{(2)} a_1^{(2)} + \Theta_{12}^{(2)} a_2^{(2)} + \Theta_{13}^{(2)} a_3^{(2)})$ 이다
    - 이것은 logistic regression이다
        + 유일한 차이점은 input이 feature vector가 아니라  hiddne layer에서 계산된 결과값이라는 것이다
* $a_1^2, a_2^2, a_3^2$ 는 original features가 아니라 계산되고 학습된 features다
* 때문에 layer 1에서 layer2로 가는 mapping(즉, $a^2$ features를 만들어내는 연산)은 다른 parameters의 집합 $\Theta^1$ 에 의해 결정된다
    - 때문에 neural network는 original input features에 제약을 받기 보다는 logistic regression을 제공하기 위한 독자적인 features를 학습할 수 있다
    - $\Theta^1$ parameters에 따라 흥미로운 것들을 학습할 수 있다
        + final logistic regerssion 연산에 사용할 모든 features를 학습할 수 있는 유연성
            + 이전의 logistic regression과 비교를 하자면 어떤 것을 분류하거나 표현할 수 있는 최선의 방법을 정의하기 위해 features를 계산해야한다
            + Hidden layers가 그렇게 하도록 하고 hidden layers에 input value를 줬을 때 어떻게 해야 output layers에 더 좋은 결과를 제공할 수 있는지 학습하도록 한다
* 기존에 봐온 newtork만이 아니라 다른 architectures(topology)에 대해서도 가능하다
    - Layer당 더 많은, 더 적은 Node의 개수
    - 더 많은 layers
    - 더 나아가 layer 2는 3개의 hidden units를 갖고, layer 3은 2개의 hidden units를 갖도록 하는 non-linear hypothesis도 존재할 수 있다

---

## Neural network example - computing a complex, nonlinear function of the input
* Non-linear classification: XOR/XNOR
* y = $x_1$ XOR $x_2$
* y = $x_1$ XNOR $x_2$
* XNOR = NOT(XOR)
* XOR은 둘 다 참이거나 둘 다 거짓일 때 맞는 예시이다
    - 가중치를 계산하는 것을 배우는 것이 아니라 NNs가 어떻게 동작하는지에 대해서 볼 것이다

### Neural Network example 1: AND function 
* 하나의 unit만으로도 logical AND fucntion을 구현할 수 있는가
    - 가능하다
* $h_\Theta(x) = g(-30 + 20x_1 + 20x_2)$
    * $\Theta_{10}^1 = -30$
    * $\Theta_{11}^1 = 20$
    * $\Theta_{12}^1 = 20$

* input에 따른 4가지 경우의 수는 다음과 같다

|$x_1$ |$x_2$ |$h_\Theta(x)$     |
|------|------|------            |
|0     |0     | $g(-30)\approx 0$|
|0     |1     | $g(-10)\approx 0$|
|1     |0     | $g(-10)\approx 0$|
|1     |1     | $g(10)\approx 1$ |

### Neural Network example 2: NOT function 
* $h_\Theta(x) = g(10 -20x_1)$

|$x_1  |$h_\Theta(x)$|
|------|------|
|0     |$g(10)\approx 1$ |
|1     |$g(-10)\approx 0$ |

* Negation은 부정을 취하려는 변수 앞에 큰 음수값을 곱하는 것으로 동작한다

### Neural Network example 3: XNOR function 
* 어떻게 XNOR Function이 동작하도록 만들 수 있는가
    - XNOR은 NOT XOR의 줄임말이다
* AND, OR, Neither 이것들을 조합하여 neural network를 만들 수 있다
![Alt text](figs\fig8-6.png)

* 1차식을 여러번 더해도 결국에는 1차식이다
    - Activation function이 없다면 결국에는 1차식만을 표현할 수 있다

---

## Multiclass classification
* Multiclass classification은 두개 이상의 카테고리로 분류하는 것이다
* 필기 인식 문제가 그 예시이다 - 10개의 카테고리가 존재한다(0-9)
    - 어떻게 해야되는가
    - One or All classification을 확장하면 된다
* 보행자, 자동차, 오토바이, 트럭을 인식하는 문제
    - 4개의 output unit을 가진 neural network를 만든다
    - output vector는 다음의 4개일 것이다
        + 1 : 0/1 보행자
        + 2 : 0/1 자동차
        + 3 : 0/1 오토바이
        + 4 : 0/1 트럭
    - 주어진 사진이 보행자라면 [1,0,0,0]을 얻을 것이다

