# Python Deep Learning



### 목차

1. Multi-Layer Perceptron(ANN)

2. Error Backpropagation

3. Optimization Method

4. Keras TensorFlow



지도학습

5. Deep Neural Network
   
- ML의 Regression, Classification
  
6. Convolutional Neural Network

   - 이미지 처리(대부분 분류 문제)

   

시계열분석(온도, 주가 등)

7. Recurrent Neural Network
   - 최근 자연어처리에 사용되지 않는다 (embedding 형식 사용)
   - 음성인식

8. Long Short-Term Memory



지도학습, 비지도학습의 개념이 조금 포함됨

9. Generative Adversarial Network

10. NLP, 추천시스템





## 1. Multi-Layer Perceptron(ANN)

>- Perceptron, 퍼셉트론이란?
>  - Artificial Neural Network(ANN, 인공신경망)의 한 종류
>- 신경망은 생물학적인 것
>
>
>
>- Multi-Layer란?
>  - Layer가 여러개
>  - 지금까지 우리가 쓴 것은... single-layer였을까?



신경망도, ML의 지도학습 알고리즘 중 하나 (input, output 필요)

함수형 모델이 발전된 형태 (GD를 이용해 w, b 학습)

발전하여 Deep Learning으로 작동





### 1) Neural Network

- 인공신경망(Artificial Neural Network)
  - 머신러닝 분야에서 연구되는 학습 알고리즘
    - 최적의 w,b를 업데이트하는 알고리즘
  - 수치예측, 범주예측, 패턴인식, 제어분야에 응용
- 인관의 뇌 구조를 모방하여 만들어짐(신경세포: Neuron)
  - 1) 수상돌기(Dendrite), 3) 시냅스(Synapse), 4) 신경세포체(Soma), 2) 축색(Axon)
    - 1) 수상돌기: 입력(x)
    - 2) 축색: 출력(y)
    - 3) 시냅스: 가중값을 갖는 연결 네트워크(w)
      - y = w1x1 + w2x2 + w3x3 + b
    - 4) 신경세포체: 노드
      - 단말기(Terminal)간을 연결하는 것이 Node
  - w가 플러스라면, 자극이 계속 크게만 전달된다
  - 우리 몸도 자극이 일정 수준을 넘으면 더이상 크게 느껴지지 않음(역치)
  - 자극을 일정 수준 안에서 조절하려고 함(sigmoid)

- 인간 뇌의 동작원리를 수학 함수로 만들어놓은 것이 "인공신경망"이다
  - f(sum(xiwi + b)), 다중회귀의 형태
  - Activation function는 보통 sigmoid를 쓴다, 다양한 함수가 존재
- input x1, x2, x3 ... 모두 더하고, b를 더해 함수를 씌우는 것



- 사람이 하는 지적인 작업들을 인공신경망을 통해 해보겠다는 것



### 2) Perceptron

> 여러가지 종류로 만들어지는 인공신경망 종류 중 하나
>
> Activation Function을 sigmoid를 쓴다



- 인공신경망의 한 종류(선형분리기)
- 가장 간단한 형태의 1) Forward Network
  - 1) 네트워크는 방향이 있다 - forward, backward
- 1957년 프랑크 로젠블라트(Frank Rosenblatt)에 의해 고안
  - 그 당시 컴퓨터가 없어서 하드웨어로 구현
  - 그런데 왜 소프트웨어 방식을 쓰는가? 싸니까!
- 동작원리
  - 노드의 입력값(Input)과 가중치(Weight)의 곱을 모두 합함
  - 합한 값이 1) **활성화함수**의 2) **임계치**보다 크면 1, 작으면 0을 출력
    - 1) sigmoid()
    - 2) 일반적으로 0.5를 줌
- 당시 컴퓨터는 빠른 대량의 산술연산을 하는 것
  - 사실 컴퓨터는 더하기밖에 못한다, 처음 컴퓨터가 나왔을 때 "adder"라고 불렀다
- 사람의 뇌는 옳고그름을 판단 - 논리연산(T/F)
- perceptron의 등장으로 사람이 아닌 기계장치가 논리연산이 가능하도록 학습시킬 수 있게 됨



#### Logic Gate

#####  AND/OR

- y = WX + b
- 통상적으로 w,x가 2개 이상이므로 대문자로 표기(행렬임을 의미)

##### NAND: AND의 부정

- 둘다 거짓인 경우에만 참
- 둘다 참인 경우에만 거짓

##### XOR: Exclusive OR

- 두개가 같으면 거짓
- 두개가 다른 값이면 참



- 퍼셉트론을 사용하면, 논리게이트를 학습시킬 수 있게 된 것이 인공지능의 큰 사건
- 대량의 산술연산에서 논리연산을 학습시켜 가능하게 함



---

인공신경망이라는 것은,

우리 머릿속의 신경망 자체를 수학적인 함수로 표현한 것일 뿐이다

그것(Perceptron)을 학습시켜, 산술연산 뿐만 아니라 논리연산이 가능하게 되었다!

"인공지능의 한 획을 그은 사건"

---



- XOR 이슈에 빠지게 됨
  - 인공지능의 빙하기에 들어가게 되는 계기
- 우리 머릿속에도 신경세포가 하나 들어있는 것은 아니듯이, 인공신경도 하나만 써서는 복잡한 문제를 해결할 수 없다
- Layer의 개념이 등장하게 됨



- 지금까지 우리가 본 것은 single "function"이었다

  - Boost는 layer의 개념처럼 볼 수도 있다

  - stacking: 머신러닝에서도 이러한 기법들이 있다(LR -> L1 -> L2 ...)

    - 머신러닝에서 stacking하는 것보다는 딥러닝하는 것이 낫다

    

- NN(Perceptron)을 하나만 쓰는 것이 아니라 여러개를 써보자(P1 -> P2 -> P3)





### 3) Multi-Layer Perceptron

> - Hidden layer가 하나
> - Input layer는 함수는 아님. 머신러닝의 feature의 개념



- y = WX + b
- 쌓았더니, XOR 문제가 풀림

- single perceptron으로 해결할 수 없는 복잡한 문제가 풀리게 됨

- 많이 쌓으면 당연히 capacity 가 좋아진다 (parameter가 많아지므로)





#### Linear Model

- 지금까지는 하나의 모델을 잘 만들려고 노력한 것



#### Nonlinear model

- 여러개의 모델을 쌓아 훨씬 복잡한 문제를 해결하려고 함

- 선 하나하나가 파라미터
- 정확히는 함수가 "노드"



- Hidden Layer의 노드를 늘릴 수는 있지만, Hidden Layer의 깊이를 늘려 하이퍼파라미터를 경사하강으로 학습시키는 것은 불가능하다고 생각했다
- 컴퓨터 기술의 발달로 Hidden Layer를 늘리는 것이 가능해짐





#### Multi Layer Perceptron(MLP)

- 다층퍼셉트론: 퍼셉트론으로 해결할 수 없는 **비선형 분리문제**에 필요

  - Hidden Layer가 하나인 신경모델을 다층 퍼셉트론 이라고 한다

- 여러 층(Layer)의 1) 퍼셉트론(Node)을 쌓아서 동작(Input Layer, Hidden Layer, Output)

  - Hidden Layer가 몇개인 게 좋은가? 해봐야 안다(Hyperparameter)
  - 늘어날수록 파라미터의 개수가 늘어난다
  - 1) Perceptron, Node, sklearn에서는 Unit (명칭은 다르지만 모두 하나의 함수라고 생각하자)

  

- Tensor

  - Multi Dimensional Matrix (다차원 행렬)

    def Machine(x1, x2,

    ​						w1_11, w1_12, b1_1,

    ​						w1_21, w1_22, b1_2,

    ​						w2_11, w2_12, b2_1) :

    ​	y1 = sigmoid(x1 * w1_11 + x2 * w1_12 + b1_1)

    



- 딥러닝의 모델은 feature를 학습한다
- 뒷쪽의 오차를 줄여가며 feature selection의 기능이 나타날 것
- 컴퓨터의 사이즈가 문제인 것
- 빅데이터냐 아니냐의 기준 - 내가 가진 컴퓨터의 사이즈
- 그것을 극복하는 방법이 "클라우드 컴퓨팅"



파라미터 개수가 증가하면 학습시간이 늘어난다





#### Binary 와 Categorical

- 예측(Regression)이라면 하나만 있어도 된다(숫자 하나만 나오면 됨)
- 이진분류(Binary)라면 y는 몇개가 필요한가? output은 하나만 있으면 된다 (0, 1 하나만 예측)
- 다중분류(Categorical)라면? 1000개를 분류해야 한다면 output 1000개
- output layer는 우리가 정하는 것이 아니라, 데이터에 따라 정해져 있다



ex) Fully-connected(FC) 연결망 (Dense)

- 3개 분류, hidden layer의 node 5개

- 5 * 3, sigmoid() - 3개의 y_hat이 각각 0~1 사이 값이 나올 것

- y의 값은 3개... 정수인코딩을 한다면 0,1,2가 나와야 함
- 그러나 sigmoid는 0~1사이 값만 나오므로 정수인코딩을 해서 사용할 수는 없다
- 이것을 0~1 사이로만 나오도록 one-hot-encoding하는 것
  - [1,0,0] [0,1,0] [0,0,1]

- 이것을 위해 sigmoid가 아니라 softmax() activation 을 사용하게 됨





### 4) softmax()

- 다중분류를 sigmoid 보다 효율적으로 처리하기 위해 처리하는 활성화함수
- sigmoid: 각각의 output이 0~1사이
- softmax는 전체의 y_hat값이 0~1 사이 값으로 나오도록 조절



- 전체확률로 개별확률을 나눠줌 (전체에서의 비율)



#### sigmoid() vs. softmax()

- sigmoid(): 함수의 출력값이 각각 0~1사이의 값을 가짐
  - 하나의 값이 높아진다고 해서 다른 값이 내려간다는 것을 장담할 수 없음(각각이 0~1 사이 값이 나오면 되므로)

- softmax(): 전체 출력값의 합이 1이 되어야하기 때문에 학습효과가 증가
  - 전체값의 합이 1이 되어야하므로 누군가의 값이 올라가면 누군가는 내려가야 한다
  - 다중분류를 한다는 것은 여러개 중 하나만 골라내는 것
  - softmax를 쓰는 것이 sigmoid를 쓰는 것보다 분류문제에서는 효율적





이진분류를 한다면, 마지막 함수에 sigmoid activation을 쓰면 처리 가능하다

그러나 다중분류의 문제를 한다면, output layer에 sigmoid 보다 softmax 를 주는 것이 효율적이다





- 이진분류에서는 y값이 0,1
- 다중분류에서는 y값은 무조건 1 (위치의 차이일 뿐, y는 1이다)
- 따라서 출력으로 나오는 y_hat도 [ , , ] 형태여야 함



#### Categorical Classification

- CEE = -sum(y * log(y_hat))
  - y는 1이므로, 모두 더해줌
- MSE VS. CEE
  - y1 = [1, 0, 0] , y2 = [0, 1, 0] , y3 = [0, 0, 1]
  - y_hat = [0.1 , 0.7 , 0.2]
  - MSE = ((0 - 0.1)^2 + (1 - 0.7)^2 + (0 - 0.2)^2) / 3
  - CEE = -0\*log(0.1) -1\*log(0.7) - 0*log(0.2)
  - MSE는 모든 경우를 계산하여 나눠줘야 하므로 계산량이 많고, CEE를 사용하면 1이 단 하나이므로 나머지는 계산할 필요가 없고 하나의 1만 계산하면 된다 (계산량이 확 줄어듦)
  - 분류문제에서는 MSE를 사용하지 않고 CEE를 사용하는 것이 훨씬 효율적이다







#### ML/DL Modeling 

1. Data Set Size가 클수록 좋다: X(input)
2. Parameter(W, b)의 개수가 많을수록 좋다



- 모델 학습성능에 부정적 영향. 왜?
- 학습의 원리가 Gradient Descent







### Deep Neural Network(DNN)

> Hidden layer가 두개 이상으로 확산







## 2. Error Backpropagation

> Backpropagation(역전파)
>
> 앞에서 오는 것은 Forwardpropagation(순전파)라고 한다
>
> Chain rule 과 역전파 알고리즘을 통해 미분값을 얻게 됨
>
> 오차 역전파와 Chain Rule을 합치면 빠르게 경사값을 획득할 수 있게 된다



---

#### 오차역전파의 궁극적인 목적

**수치미분 과정 없이 경사하강을 위한 미분값 획득**

1. 신경망 모델: 간단한 함수의 합성(중첩) 함수
   - 합성함수의 미분은 구성하고 있는 개별함수의 미분의 곱 (Chain Rule)

2. 전방향 연산에서 계산된 값을 재사용
3. 전방향 연산에서 계산된 Error값을 전달

---





순전파에 들어가는 것은 Input

뒤에서는 뭐가 올까? Error



- 함수형 모델의 학습원리: Gradient Descent
- 경사값(dw)을 계산하는 데 시간이 오래걸림(컴퓨터 능력이 많이 필요해짐)
- 시간이 오래걸리는 문제를 해결하기 위해 (결국 미분값이 필요함)
- 에러(y - y_hat)를 뒤쪽으로 보내면, 미분을 하지 않더라도 학습이 된다





### 미분의 기본적 개념

- 미분: 순간변화량 or 접선의 기울기 (전향, 후향)

  - 함수: X(input)와 y(output)의 관계 / y = f(x) = ~~

  1. 입력변수(X)가 변할 때, 함수 f(x)의 변화량
  2. 함수 f(x) (=y) 가, 입력변수(x)에 얼마나 민감하게 반응하느냐 



- x -> x + delta x

- x: delta x, y: f(x + delta x) - f(x)

- f(x) = df(x) / dx

  = lim (f(x + delta x) - f(x)) / delta x : delta x가 0으로 수렴할 때, y가 움직인 값 / x가 움직인 값



#### 미분하면, 

| f(x) = a (a = 상수) | f(x) = 0            |
| ------------------- | ------------------- |
| f(x) = ax^n         | f(x) = n * ax^(n-1) |
| f(x) = e^x          | f(x) = e^x          |
| f(x) = ln^x         | f(x) = 1/x          |



#### 편미분

- W, b가 함께 바뀌기 때문에 편미분이 필요하다

- 미분과의 차이점

  1. 입력변수가 2개 이상인 다변수 함수의 미분

  2. 변수 1개를 제외한 나머지 변수들은 상수로 취급하고 지정된 하나의 변수에 대해서 미분을 수행

     ex1) ∂f(x, y) / ∂x  ( ∂ = 라운드 d)

     ​	= ∂(3x + 2xy + y^2) / ∂x

     ​	= 3 + 2y

​				ex2) ∂f(x, y) / ∂y

​					= ∂(3x + 2xy + y^2) / ∂y

​					= 2x + 2y

​				ex3) 체중(y) ~ 운동량(x1) + 식사량(x2)

​					운동량이 체중에 주는 영향과 식사량이 체중에 주는 영향

​					∂체중 / ∂운동량 vs. ∂체중 / ∂식사량



- 함수와 함수가 중첩되어 있는 형태로 나타나므로 (sigmoid 안에 다변수 함수가 들어있는 형태)



#### Chain Rule(연쇄법칙)

- 합성 함수 미분은 합성 함수를 구성하는 개별함수 미분의 곱으로 처리
- f(x) = e^(3x^2)
- t = (3x^2)로 치환함
- f(t) = e^t, t = 3x^2
- ∂f(x) / ∂x = ∂f(t) / ∂t * ∂t / ∂x



#### 중앙미분(차분)

- 앞쪽에서 뒷쪽으로 이동
- 뒷쪽에서 앞쪽으로 이동

- 오차를 최소화하기 위해 사용

- 가운데를 x로 보고

- 뒷쪽은 x- delta x, 앞쪽은 x + delta x

- 따라서 우리가 구하고 싶은 값은 f(x+ delta x) - f(x - delta x) / 2 * delta x 이 된다

  



### 1) Forward Propagation

> 기본적으로 신경망은 순전파로만 동작한다
>
> 앞쪽으로 가면서 뒤의 오차가 줄어들도록 학습



- Hidden layer가 깊어지면, 

- 각 학습단계의 Parameter Update를 위해서 Parameter별 편미분값 필요
- 만약 Parameter개수가 100만개 이상이라면 수치미분으로 가능한가?
  - 100만개가 결코 많은 것이 아니다. 1000만개도 훌쩍...
  - 가능하다 해도 많은 컴퓨팅 자원과 시간이 소요
  - MLP에서는 Hidden Layer와 Node의 개수에 따라 Parameter의 개수 증가
  - Parameter개수는 Model Capacity에 영향을 줌
  - 하지만 Parameter개수 증가는 학습 시간에 부정적 영향



### 2) Chain Rule

> 우리가 보고 있는 신경망 모델은, Node(Function)들의 연결
>
> Chair Rule의 핵심: 합성함수의 미분은, 개별함수의 곱으로 나타낼 수 있다
>
> 딥러닝은 어렵고 복잡한 식을 푸는 것이 아니라, 아주 간단한 수학식을 많이 푸는 것



- 신경망에서 동그라미 하나는 sigmoid(WX+b) / W,X 는 Matrix

- Hidden layer 하나를 통과하면, sigmoid(WX+b)가 새로운 input(X)이 됨
  - sigmoid(W*sigmoid(WX+b))
  - sigmoid(W\*sigmoid(W\*sigmoid(WX+b)))

- layer가 3개라면, 3개 함수의 합성(중첩)



- 미분의 연쇄 법칙(Chain Rule)을 반복적으로 적용하는 알고리즘
  - Neural Network는 간단한 함수들의 중첩(합성함수)으로 구성
  - 합성함수의 미분은 합성 함수를 구성하는 개별함수 미분의 곱으로 처리
- 전방향 계산하며 구했던 에러값을 backpropagation으로 다시 보내주면, 미분하는 과정 없이 미분값을 얻게 된다



### 3) Backpropagation Algotithm

> 결국 개별함수의 미분도 미분을 해야 알 수 있는 것



- Forward propagation을 수행하여 y_hat 계산
  - y와 y_hat을 사용하여 오차값을 계산 (MSE or CEE)
- 학습: 오차값이 감소하는 방향으로 weight(or bias)수정
  - 효율적인 경사값 계산을 위하여 Backpropagation 수행
  - Parameter Update을 위해 출력층(Output Layer)의 오차값을 은닉층(Hidden Layer)으로 전달
- 미분의 연쇄법칙 + 오차역전파 알고리즘
  - 수치미분 과정 없이 학습의 위한 경사값 계산
  - Hidden Layer나 Node가 증가해도 학습속도에 영향을 받지않음
- 오차값(y - y_hat)을 사용하여 W21, W11 Parameters Update
- W21의 미분: W21이 변할 때, y_hat이 얼마나 감소하는가? 로 표현 가능 
- WS 한 애를 SA(Sigmoid Activation) 한 output이 y_hato1



#### 연쇄법칙 적용

- Chain Rule을 사용하면, W21의 미분이 3개의 함수의 곱으로 표현 가능해짐

  - ∂Cost / ∂W21 = ∂cost / ∂y_hato1 * ∂y_hato1 / ∂WS3 * ∂WS3 / ∂w21

  - 오차값 * 출력값(1-출력값) * 입력값

  - 뒤로부터 전달받은 오차값과 전방에서 전달받은 입력값, 출력값 재사용

    

- 그러나 실제로 미분할 필요가 없다! 왜?

- ∂WS3 / ∂w21 = y_hatn1*W21 + y_hatn2W23 = y_hatn1

  - 굳이 미분하지 않아도, 그 값이 이전 노드의 Input임

- ∂y_hato1 / ∂WS3

  - y_hato1은 sigmoid함수의 output. sigmoid(WS)
  - sigmoid의 미분은 sigmoid(x) * (1-sigmoid(x)) 으로 구해낼 수 있다(증명 참조)
  - 출력값(1 - 출력값)

- ∂cost / ∂y_hato1

  - ∂Cost = (∂Costy_hato1 + ∂Costy_hato2) / ∂y_hato1
  - 오차값이 됨

- 굳이 미분하지 않아도, **Forward propagation 결과들의 조합 패턴**

  

- 은닉층이 깊어질수록 식은 길어지지만 chain rule을 통한 구성은 똑같음



역전파는 학습을 하기 위한 미분값을 주는 것이지,

학습은 여전히 경사하강법으로 이루어진다



---

#### 핵심

forward propagation 에서 구한 값과 오차를 뒤로 보내주면

미분을 하지 않아도 미분값을 구할 수 있다!

---



첫번째 이슈

1. 신경망에서 경사하강법을 쓴다는 것은 심각한 부작용이 나타난다
   - 신경망이 깊어질수록 학습이 불가능해지는 문제점들이 나타남 - 역전파 알고리즘 사용으로 극복
2. 역전파 알고리즘의 부작용





### 4) Vanishing Gradient

> "경사 소멸", "경사 소실"
>
> Gradient: Parameter 학습을 위한 미분값
>
> Vanishing: 0이 된다 -- 아무런 변화가 생기지 않는다
>
> Layer가 깊어지면 발생
>
> 신경망 초창기에는 시그모이드가 주로 쓰였지만 은닉층을 다수 사용하는 딥러닝 시대가 되면서 ReLU 가 더 많이 쓰이고 있다. (빨간책 p.43)



- Backpropagation은 학습에 긍정적 영향을 주는 것은 사실

- 그런데, 층이 늘어날수록 계속 sigmoid의 미분값을 곱해주어야 한다
  - sigmoid 값의 범위는 0~1 사이
  - sigmoid를 미분하면, sigmoid(x) * (1 - sigmoid(x)), 범위는 0~0.25사이가 된다
  - 0.25를 계속 곱해준다는 의미: 0에 가까워지게 된다



- Backpropagation의 장점은 layer를 깊게 쌓아 parameter개수를 늘리고 더 복잡한 문제를 해결할 수 있는 것인데, sigmoid의 미분값을 계속 곱해주는 문제가 발생하게 되고, 결과값이 0에 가까워져 사라지게 됨

- 오히려 parameter 학습이 되지 않는 부작용이 나타나게 됨



- 역전파는 출력층으로부터 하나씩 앞으로 돌아오면서 각 층의 가중치를 학습
  - 신경망 모델의 가중치 학습에 미분값, 즉 기울기가 필요
  - sigmoid함수를 미분하면 최대치가 0.25로 1보다 작은값 생성
  - 은닉층이 증가하면 미분값(기울기)가 0이 되는 문제가 발생
- 대응책: Activation Function을 다른 함수로 대체하여 학습
  - Tanh: 미분하면 0~1 사이
  - ReLU(Rectified Linear Unit): "정류", 0보다 작은 값이 나오면 모두 0이 되고, 0보다 큰 값은 상수이므로 미분하면 1이 됨
  - Leaky ReLU: 0이 곱해지면 안되므로 0보다 작은 값에 기울기를 준다
  - Exponential Linear Unit: 지수함수 형태로 약간의 곡선을 준다
- 뭐가 제일 좋을까? 해봐야 안다 (하이퍼 파라미터)







## 3. Optimization Method

> 우리의 모델을 주어진 데이터에 최적화하는 방법
>
> "경사하강법"
>
> 이것을 좀 다르게 접근해보면 어떨까?
>
> 어느 optimization method가 가장 좋은가? 해봐야 안다 (하이퍼 파라미터)



### 1) Stochastic Gradient Descent(SGD)

> 노란책 p.83
>
> 확률적 경사하강(경사하강을 좀 더 발전시켜보자)
>
> 통계적 개념: 일부 데이터로 전체 데이터를 추정

- 전체데이터(batch) 대신 일부 데이터(mini-batch)를 사용 (랜덤 샘플링)
  - 100개를 10번씩 10번 (한번 할 때마다 parameter update)
- batch GD 보다 부정확할 수 있지만 **계산속도가 훨씬 빠름**
- 같은 시간에 더 많은 step을 이동 가능
- 일반적으로 batch의 결과에 수렴
  - 표준집단의 평균을 평균을 내면 모평균의 평균에 수렴하는 것과 비슷함
- 현재 다양한 SGD의 변형이 존재





### 2) Momentum / Nesterov Momentum(NAG)

> 이동방향에 가중치를 주어 급격하게 꺾이는 것을 막는 방식



#### Momentum

- 이동과정에 '관성'을 반영
- 현재 이동하는 방향과 별개로 과거 이동 방향을 기억하여 이동에 반영
- 이전 가중치의 업데이트양(벡터)과 방향이 크게 변화하지 않도록 조정
- 바로 직전 시점의 가중치 업데이트 변화량을 적용 (벡터연산 적용)
- W(t+1) = W(t) - r*dE + µ\*dW





### 3) Adaptive Gradient(Adagrad) / RMSProp

> 학습률을 감쇠시키는 방식



#### Adaptive Gradient

- 학습률이 작으면 안정적이지만 학습속도는 느려짐
- 학습횟수가 증가함에 따라 **학습률을 조절**하는 옵션 추가
- 최솟값에 다가갈수록, 이동거리가 작아질수록 최소점에 안착할 확률이 높아짐
- 변하지 않던 학습률을 학습횟수에 따라 줄어들게 만드는 방식

- 학습률 감쇠식 추가
  - r = r / (1 + p*n) n: 학습횟수
  - g = g + (dE)^2
- 경사하강 식 자체가 달라짐



#### RMSProp

- Adagrad의 단점인 Gradient 제곱합을 지수평균으로 대체





### 4) Adaptive Moment Estimation(Adam)

- RMSProp과 Momentum 방식의 **장점을 합친** 알고리즘
- Momentum과 같이 지금까지 계산해온 기울기의 지수평균을 저장
- RMSProp과 같이 기울기 제곱값의 지수평균을 저장



---

- 기본적으로 batch size는 다 적용이 되어있고,

- 이동거리를 변화시킬 것이냐, 학습률을 변화시킬 것이냐, 다 쓸 것이냐의 문제

- 최근은 momentum만 적용하는 경우는 거의 없고, RMSProp이나 Adam을 사용한다

  (학습률의 감쇠를 가져가면서 momentum을 추가하는 추세)



최적화 기법은 아직도 연구중이다

---







우리가 이야기하는 딥러닝 모델이라는 것은,

파라미터들의 집합이다 A set of parameters

파라미터들이란? 노드 간 연결되는 선

함수는 정해져있다  y = A(WX+b)

사실은 모델을 저장하는 것은, 모델 구조와 파라미터 값만 가지고 있으면 된다 (텍스트 정보임)

파라미터 정보를 저장해뒀다가 그 숫자를 다시 사용하는 것



양자 컴퓨팅이 상용화되면 딥러닝의 발전은 매우 빨라질 것!







## 4.  Keras TensorFlow



### 1) TensorFlow

- 데이터 흐름 프로그래밍을 위한 오픈소스 소프트웨어 라이브러리
- Neural Network같은 Machine Learning 프로그램에 활용
- 구글 브레인 팀에 의해 개발, 2015년 아파치 2.0 오픈소스 라이센스로 공개
- 주요 특징
  - Keras API를 활용하여 손쉬운 모델 빌드
  - 플랫폼 관계 없이 모델을 학습시키고 배포 가능
  - 빠른 프로토타입 제작과 디버깅 구현



딥러닝 모델링이 매우 쉬워지며, 

2.x 버전이 되면서 많은 기능이 추가되고 있다



### 2) Keras

> Keras: 그리스어로 "뿔"이란 뜻
>
> - 실제 케라스가 모델링하는 것이 아니라 TensorFlow로 모델링
> - 꿈을 관장하는 신이 상아로 만들어진 문으로 꿈을 내보내면, 꿈이 이루어지지 않음
> - 뿔로 만들어진 문으로 꿈을 내보내면 꿈이 이루어짐(예지몽)
> - 케라스를 통해 예지몽과 같은 모델을 만들겠다는 철학적인 의미!
>
> Keras 자체로는 아무것도 할 수 없다
>
> TensorFlow와 대화하는 것을 보다 쉽게 해주기 위한 인터페이스라고 생각하자
>
> R-R Studio, Python-Jupyter Notebook의 관계와 비슷



- Python기반의 Deep Learning Framework(library)
- 내부적으로는 TensorFlow, Theano, CNTK등의 Deep Learning 전용 엔진 구동
- 누구나 쉽게 Deep Learning Model 생성 가능
- Keras 사용자는 복잡한 내부엔진에 대하여 알지 못해도 됨
- 직관적인 API를 통하여 MLP, CNN, RNN등의 모델 생성 가능
- 다중 입력 및 다중 출력 구성 가능
- TensorFlow 1.14버전부터 공식 코어 API로 추가



- 사용자 중심의 1) 상위 레벨 인터페이스 제공
  - 1) 상위레벨: 유저와 가까움(고급언어) / 하위레벨: 하드웨어(컴퓨터와 가까움)(저급언어)
  - 하위 레벨 계산은 일반적으로 TensorFlow 사용
  - 동일한 코드를 CPU 및 다양한 GPU에서 실행 가능



#### Keras with GPU

> 박사 8명 있는 것보다 초등학생 2천명 있는 게 낫다

- CPU(Central Processing Unit): 복잡한 연산 수행에 적합
  - 프로세서: 8코어(CPU가 8개)
- GPU(Graphic Processing unit): 단순한 **대량 연산**에 적합
  - Deep Learning Matrix 연산에 활용
  - NVDIA 기준 코어가 2944개, 최근은 4천개 대 코어가 등장



### 3) Tensor

> 다차원 행렬
>
> 텐서라 부르는 다차원 넘파이 행렬. 핵심적으로 텐서는 숫자 데이터를 위한 컨테이너(container)입니다. 항상 수치형 데이터를 다루므로 숫자(float)를 위한 컨테이너입니다.(노란책 p.61)



- Rank1 Tensor - 벡터

- Rank2 Tensor - 행렬(Matrix)

- Rank3 Tensor

- Tensor는 텐서플로로 만들어도 되고, numpy로 구현해도 상관없다(casting 필요)



- Neural Network 학습의 기본 데이터 단위
  - 숫자(Numeric) 데이터를 담기 위한 컨테이너
  - 임의의 차원(Dimension) 또는 축(Rank)을 가짐



- Type

| Rank | Type            | Example                                                     |
| ---- | --------------- | ----------------------------------------------------------- |
| 0    | Scalar          | 0                                                           |
| 1    | Vector          | [0, 1]                                                      |
| 2    | Matrix          | [[0, 1] , [1, 0]]                                           |
| 3    | 3 Tensor(Array) | [[[0, 0] , [0, 0]] , [][1, 1] , [1, 1]] , [[2, 2], [2, 2]]] |
| N    | N Tensor        |                                                             |





#### Tensor in NLP(Natural Language Processing)

> 사람이 이해하는 언어를 숫자로 바꿔주어야 컴퓨터가 이해할 수 있다

- 문장과 단어를 숫자 벡터로 매핑

  ex) 하얀 고양이, 하얀 강아지, 하얀 비둘기

  - 단어 단위로(영어라면 캐릭터 단위로) 나눠서 처리, 중복되지 않게 단어 단위로 숫자

- Unique Word Dictionary(Rank1)

  | Word   | Index | One-Hot Encoding Vector |
  | ------ | ----- | ----------------------- |
  | 하얀   | 0     | [1, 0, 0, 0]            |
  | 고양이 | 1     | [0, 1, 0, 0]            |
  | 강아지 | 2     | [0, 0, 1, 0]            |
  | 비둘기 | 3     | [0, 0, 0, 1]            |

  - ex) 하얀 고양이, 하얀 강아지, 하얀 비둘기(Rank2)

    - 하얀 고양이: [[1, 0, 0, 0] , [0, 1, 0, 0]]
    - 하얀 강아지: [[1, 0, 0, 0] , [0, 0, 1, 0]]

  - mini-batch Input(Rank3): 문장으로 넣고싶다면 보다 복잡한 텐서로 입력하게 된다

    - 하얀 고양이, 하얀 강아지, 하얀 비둘기

      [[[1, 0, 0, 0] , [0, 1, 0, 0]] , [[1, 0, 0, 0,] , [0, 0, 1, 0]] , [[1, 0, 0, 0] , [0, 0, 0, 1]]]

    - shape로 모양을 확인하면 (3, 2, 4)

  - 어떻게 묶느냐에 따라 다양한 형태의 Tensor로 구현할 수 있다



#### Tensor in Grayscale Image(흑백이미지)

- 색을 표현하기 위해 0~255 (하얀색 255, 검은색 0)

- 이미지는 자연어보다 좀 편하다(이미지는 이미 숫자들의 컨테이너, Rank2 Tensor)
- 이미지를 여러개씩 집어넣어 처리(Rank 3 Tensor)
  - (Number of Images, Rows, Columns)
  - (3, 5, 5) : "5*5 이미지 3장이 들어갈거야!"



#### Tensor in RGB Color Image

> 노란책 p.68

- (Number of Images, Rows, Columns, RGB Channel)
- (3, 5, 5, 3)
- 컬러이미지는 기본적으로 3차원의 구조를 가진다
  - R
  - G
  - B
  - 세개가 묶여 하나로 보이는 것
  - R 255, G 0, B 0 이면 빨간색, 세개 모두 255면 하얀색, 0이면 검은색
  - #을 붙여 16진수로 표현함
- 디지털의 모든 색은 RGB 숫자로 표현되고 있다 (Rank 3 Tensor로 저장하고 있음)

- 처리하는 픽셀의 개수가 몇개냐에 따라서 1000만 화소, 2000만 화소 (들어있는 숫자의 집합) 등으로 표현하는 것
- 4K(4096 * 4096)



#### Tensor in RGB Color Video

> 컬러이미지는 컬러이미지의 집합

- 영상은 "프레임"의 개념이 들어감(초당 몇장의 그림이 흘러갈 것이냐?), 많을수록 자연스러워짐
- (3, 5, 600, 800, 3)
- (Video Frames, Number of Images, Rows, Columns, RGB Channel)
- 프레임당 들어있는 이미지의 수
- 프레임을 한번에 여러개 처리하고 싶다면 5 Rank Tensor가 된다



Tensor는 정해져있는 것이 아니라, 어떻게 처리할지 우리가 다룰 모델에 맞게 넣어줘야 한다





### 4) Keras Modeling

> Tensor(실수형 행렬)로 만들어져 있어야 한다 (Preprocessing)
>
> Modeling 전에 데이터 전처리 & EDA 작업이 필수 (실제 가장 많은 시간이 걸린다)
>
> 목표달성시까지 반복(모델구조 및 Hyperparameter 변경)
>
> Data Collection - Data Preprocessing - Modeling



#### (1) Define(모델 신경망 구조 정의)

> 모양만 만듦

- Sequential Model
- Layers/Units(=Node)
- Input Shape
- Activations: sigmoid? tanh? ReLU?



#### (2) Compile(모델 학습방법 설정)

> 어떤 문제를 풀건가? (예측/분류)
>
> Loss, Optimizaer는 계속 새로운 것이 업데이트되고 있다

- Loss: 예측(MSE), 분류(Binary CE / Categorical CE)
- Optimizer: GD에 대한 부분
- Metrics: 예측(MSE), 분류(Accruacy, Preicision, Recall)



#### (3) Fit(모델 학습 수행)

> 실제 학습이 수행
>
> Parameter Update
>
> 구조에서 정의된 파라미터가 실제로 학습됨

- Train Data
- Epochs: 반복횟수
- Batch Size
- Validation Data



#### (4) Evaluate(모델 평가)

> ML의 validation 단계 (사용에 적합한가?)
>
> 아니라면 앞단계로 돌아가며 원하는 성능이 나올 때까지 반복

- Plot: 시각적으로 봄
- Evaluate(함수로 볼 수도 있음)



#### (5) Predict(모델 적용)

> 개발단계(Development)에서 Live Environment로 Model Deployment하는 단계
>
> (실제로 모델이 사용되는 단계, 배포단계)
>
> "Go Live 한다" 고 표현 (실무용어)

- Probability
- Classes





## 5. Deep Neural Network

> 모델을 만드는 것은 어렵지 않으나, 모델에 데이터를 넣기 위한 전처리가 중요하다



### 1) Binary Classification (p.104)

> 데이터 전처리에 대한 고민

- IMDB(Internet Movie Database)

  - 영화에 대한 5만개의 양극화된 감상평 제공
  - 감상평에 포함된 단어를 기준으로 긍정과 부정으로 이진 분류
  - train 25000, test 25000

- 문제점: 리뷰글의 길이가 대부분 다르다 (비정형데이터), 어떻게 input으로 넣는가?

- 어떻게 똑같은 형태의 input으로 넣을 것인가?

  ex) 첫번째 리뷰 218개 단어(문장길이) [1, 14, 22, 16, 43, 530, 973, ...]

  ​      두번째 리뷰 181개 단어 [ ..... ]

  - 단어의 종류는 10000가지를 넘지 않음(10000등까지만 뽑았으므로)
  - 길이를 모두 10000으로 맞추고, 단어가 있는 곳에만 체크하는 식으로 데이터 변형
  - 길이가 10000인 25000개의 데이터 생성



### 2) Categorical Classification (p.56)

> 데이터 shape 변환에 대한 고민

- Handwritten Digits in the M(Mixed)-NIST Database
- 문제점: input이 2d
- 2d를 하나로 길게 펴주는 작업 필요 ex) (5, 5) - (25,)
- 데이터의 특징을 가지고 있을 것이다



- 이미지 데이터는 보통 Standardization이 아니라 Normalization을 해준다. 왜?
  - x - min / max - min
  - min은 보통 0, max는 255이므로 x / 255가 된다
- accuracy: 받아들일 수 있는가 없는가? 
-  acceptance criteria가 reasonable해야한다
- 이전에 사람이 분류했을 때 정확도는 얼마였는가?
  - 사람의 오류가 3%였다면?
  - 사람의 오류는 1.5%라면?
  - 쓸까 안쓸까? 기계를 쓰고 인건비를 줄이는 것과, 기계 오류에 따른 손해를 배상해주는 것?
- 얼마나 좋게 만들어야 하나? 기준을 잡고 들어가야 한다




강화학습으로 가면, 모든 아날로그를 디지털화 해야한다 (CPS, Cyber Physical System)
디지털 세계로부터 무언가를 학습하게 됨




#### Overfitting Issues

> 노란책 p.151

- Train 데이터에만 최적화된 상태
- 데이터가 적을 때: 데이터의 숫자에 비해 Model Capacity가 높을 때
- 파라미터의 수가 많을 때
- Capacity를 높게 잡고 낮추는 방식이 일반적인 접근 방식



##### Train Loss vs. Validation Loss

1. 더 많은 Train Data (현실적으로 어려운 경우가 많다)
2. Model Capacity가 높아서
   - 층을 줄이거나,
   - Node를 줄이거나
3. L2 Regularization
4. Dropout
5. Batch Normalization



##### (1) Model Capacity 감소 전략

- Hidden Layer 및 Node 개수 줄이기



##### (2) L2 Regularization(규제화)

- Model이 Train Data에 너무 학습되지 않도록 방해
- 가중치의 제곱에 비례하는 노이즈를 Cost Function에 추가(가중치 감쇠: Weigh Decay)
  - L1인 경우 가중치의 절대값



##### (3) Dropout

> 기존 머신러닝에는 없던 개념
>
> 경험적의 산물(실제 해봤더니 좋아지더라)

- 훈련과정에서 네트워크의 일부 출력 특성의 연결을 무작위로 제외시킴
- dropout되는 순간 capacity가 떨어지는 것 (weight의 개수가 줄어듦)



##### (4) Batch Normalization
> 모델의 성능을 향상시키기 위해서 최근 가장 많이 적용되는 것은, Batch Normalization이다 

- Scaling의 기법: Normalization, Standardization
  - Normalization: Scaling 중 Min-Max Scaler
  - Standardization과 Normalization을 합쳐 Normalization이라고 하기도 한다
  - Standardication: 표준화란 평균이 0, 분산이 1인 정규분포의 형태로 배치시키는 것
  - 용어는 Batch Normalization이지만, 내부적으로는 표준화가 동작한다
- Batch Normalization을 하면 Overfitting 외에도 긍정적인 효과가 나타난다(학습이 더 잘되는 등)

- 정규분포인 경우 예측이든 분류든 더 잘 이루어진다




- Input Data(X)가 연속형 데이터인 경우, **Activation 함수로 들어가기 전**, Batch Normalization을 처리해서 전달

  - 활성화 함수의 입력값을 정규화 과정을 수행하여 전달
  - input이 들어가면 노드에서 wx + b 처리되고, 활성화 함수를 씌워 다음 노드로 넘어간다
  - 정규분포 형태로 들어가도, wx + b를 거치면 정규분포의 형태로 나오지 않는다
    (한쪽으로 치우치거나 알수없는 분포로 나올 수 있다)

  - 나온 output을 정규분포 형태로 만들어서 활성화함수에 전달한다
- layer를 지날 때마다 분포가 변하므로, 다시 정규화한다




- Gradient Vanishing 문제 해결 및 더 큰 Learning Rate을 사용 가능(학습이 빨라진다)
- 이론적 이유: Sigmoid를 기준으로 설명
  - Sigmoid를 나오면 기본적으로 0, 1 값이 나옴
  - Activation에 들어가는 분포가 한쪽으로 치우쳐져서 들어간다면, 0에 가까운 값으로 떨어지는 것을 줄여준다
  - 한쪽으로 기운 데이터라면 0에 가까운 값이 나올 가능성이 크지만, 정규분포라면 그보다 큰 값이 나올 확률이 더 커진다
- capacity를 줄이거나, noise를 키우는 것이 아니라 중간값들이 극단값으로 가지 않도록 조정해주는 것


- Batch Normalization에서 Parameter가 들어간다. 
  - Parameter의 의미: 정규분포 형태로 만들기 위해 평균, 분산이 필요. 어떤 평균과 분산이 최적일지 학습시키기 위한 Parameter






### 3) Regression Analysis

> 빨간책 p.91, 
>
> 노란책 p.165 (테이블 참조)
>
> 예측모델은 딥러닝을 잘 사용하지 않는다(머신러닝으로 충분히 가능)

- Boston Housing Price Dataset
- 1970년대 중반 보스턴 교외 지역의 주택 평균가격 예측
- 정형화된 데이터가 주로 쓰이고, 머신러닝으로도 보통 가능하다



- 학습 횟수(epoch)도 Model Capacity에 영향을 준다
- 학습을 많이 했더니 오히려 성능이 안좋아진다 (overfitting 발생)
- 분류문제와 달리 예측모델인 경우에는 MAE 자체가 적은 것이 성능이 좋은 모델이라고 할 수 있다
- 500번이나 epoch를 돌 것 없이, 80-90번만 돌면 parameter가 최적이 된다
- overfitting을 줄이기 위해서 학습을 적게 시키면 된다
- 그런데 만약 1000번을 학습하면, 올라갈까 내려갈까? 연구결과 일반적으로 내려가지 않는다
  - 횡보하거나 좀 더 올라가는 것이 일반적
- 만약 오차가 계속 줄어들고 있는 상태라면? 학습을 더 시켜야 한다
- 더 이상 좋아질 것 같지 않으면 학습을 그만시키면 좋지 않을까? Early Stopping, model checkpoint, callback



#### Early Stopping

> 노란책 p.331

- Keras Callback의 두가지 기능을 이용해 가능함
  - EarlyStopping()
  - ModelCheckpoint() : 최소값이 나타날 때마다 모델을 저장





### ***Hyperparameter Optimization***

>  DNN은 Hyperparameter 튜닝을 어떻게 하느냐에 따라 달려있다
>
>  기억하자! 이것이 나의 능력

- Layer & Node & Input_shape
- Activation Function
- Loss & Optimizer
- Epoch & Batch_size
- Kernel_regularizer & Dropout
- K-fold Cross Validation
- Train vs. Validation Rate & Normalization & Standardization





### DNN의 한계점

이미지처리: Rank3 에서 Rank1 Tensor로 변환하는 순간, 개체가 가지고 있는 위치정보가 소멸됨

2차원으로 변환하면서 위치정보까지 손상되지 않아야 다음단계 일을 할 수 있다

Deep Learning 모델은 1차원만 받을 수 있고, Input은 2차원인데 어떻게 해야할까?







## 6. Convolutional Neural Network

> 이미지를 이미지 그대로 처리하고 싶다
>
> 이미지 특징을 뽑아내면서 위치정보가 손상되지 않게 하고싶다
>
> CNN + DNN
>
> "Convolutional"이 핵심
>
> 이렇게 연산하므로써 얻어지는 이점은 무엇인가?
>
> CNN은 이미지에만 쓰이는 것은 아니다. 
>
> 음성, 자연어에도 사용됨





### 1) 합성곱(Convolutional) 신경망 알고리즘

- 이미지 처리 작업에 주로 사용
  - 반드시 이미지처리에만 사용하는 것은 아니다
- **합성곱 연산**을 이용하여 **가중치의 수를 줄이고 연산량을 감소**
  - Model Capacity를 떨어뜨리는데 어떻게 더 잘되나?
- 여러개의 **Filter(Parameter Matrix)**로 이미지의 특징(Feature Map)을 추출
  - 하나가 아니라 여러개의 필터를 쓴다
  - 필터에 적용되는 하이퍼파라미터: 필터의 개수, 필터의 사이즈
  - 우리가 학습시킬 대상이 Filter이고 Parameter가 matrix 형태로 생김
  - feature extraction을 통해 feature map을 만듦
  - 기존에는 파라미터들이 
- 1) Local connectivity & 2) Paramter Sharing
  - 1) 주변의 픽셀과의 관계가 망가지지 않음
  - 2) 하나의 필터(가중치)로 이미지에 고르게 흐르면서 필터를 공유해서 쓰는 개념으로, 가중치의 ㄱㅐ수를 줄일 수 있게 됨
- 말단(Top)에 sigmoid 또는 softmax함수를 적용하여 이미지 분류작업 수행
- LeNet, AlexNet, VGGNet, InceptionNet, ResNet 등으로 발전
  - 이미 잘 만들어진 기성모델들이 많다 (가져다 쓰는 것이 더 성능이 좋은 경우가 많음)



### 2) CNN에 추가된 Hyperparameter

- Filter

- Stride

- Pooling

- Padding

은 반드시 기억하자!



#### (1) Filter

> CNN에서 가장 중요한 Hyperparameter
>
> 크기, 개수
>
> 필터가 파라미터이다! (학습하면서 계속 바뀐다)
>
> 통상 필터의 크기는 홀수로 준다(3,5,7)



> 빨간책 p.139
>
> 수작업으로 설계한 특징에는 세 가지 문제점이 있습니다. 첫쨰, 적용하고자 하는 분야에 대한 전문적 지식이 필요합니다. 둘째, 수작업으로 특징을 설계하는 것은 시간과 비용이 많이 드는 작업입니다. 셋째, 한 분야에서 효과적인 특징을 다른 분야에 적용하기 어렵습니다.
>
> 딥러닝 기반의 컨볼루션 연산은 이런 문제점들을 모두 해결했습니다. 컨볼루션 신경망은 특징을 검출하는 필터를 수작업으로 설계하는 것이 아니라 네트워크가 특징을 추출하는 필터를 자동으로 생성합니다. 



> 노란책 p.170 ~ 175, 177



이미지 위에 필터를 올려놓고,

- Filter를 Input_Data에 적용하여 특징 맵(Feature Map) 생성

- Filter의 값은 Input_Data의 특징(Feature)을 학습하는 가중치 행렬

- 동일한 Filter로 Input_Data전체에 합성곱 연산(Convolutional) 적용
  - input(image) - filter(convolutional) - output(feature map)
  - 이미지의 특징을 여전히 가지고 있을 것(이미지의 특징이 추상화됨)
  - 위치정보가 망가지지 않음

- 이미지가 한번 filter를 거치면 이미지가 아니라 feature map이라고 부른다

- convolutional layer를 몇번 하는 게 좋은가? 해봐야 안다

  

ex) 120 * 160 이미지

- filter = 5*5
- 116 * 156 * 12가 됨. 의미? 120 * 160 이미지를 5*5 필터 12개로 처리함

- 풀링을 거치면 사이즈가 절반으로 줄어듦 58 * 78
- 2 * 2 맥스풀링: 2 *2 에서 가장 큰 숫자만 뽑아냄



- 처음에는 필터에 랜덤값이 들어감
- Feature Map으로 y_hat을 구하고 y와 비교. CEE를 줄이기 위해 역전파, parameter update
- Filter의 값이 학습됨
- 옛날에는 Filter 값을 사람이 고정으로 넣었음
- 스스로 학습할 수 있도록 filter 값을 parameter set으로 사용. CEE가 줄어들도록 GD에 의해 update 됨
- 데이터를 주고, 데이터의 특징에 맞는 filter를 찾도록 함(여러 필터를 쓰는 이유)
  - 필터마다 세로 특징, 가로 특징, 곡선 특징 등을 찾아냄

- 필터 parameter matrix에 들어가는 값들이 모두 파라미터 값이고, 이 파라미터가 이미지의 특징을 표현하도록 업데이트 된다



- 필터를 여러개 주나, 모든 필터가 동일한 패턴을 학습하지 않을까? 그렇지 않다.

  - 필터의 초기값이 같다면 다 동일하게 학습됨
  - 랜덤값을 가지게 되면, 필터가 다 다르게 학습이 된다(그래야 오차가 줄어들므로)
  - 많은 필터를 주어야 많은 패턴을 학습한다
  - 목적은 filter값을 찾아내는 것이 아니라, 이미지의 패턴이 실제 정답과 가깝도록 feature map을 뽑아낼 수 있는 filter 값으로 바뀌어야 하는 것이므로

  



#### (2) Stride

> 통상적으로 별로 건드리지 않는다. 한칸씩 움직이는게 일반적으로 더 좋더라.

- (1, 1)
- Filter를 적용하기 위해 이동하는 위치의 간격
- Stride값이 커지면 출력 특징 맵(Feature Map)의 크기가 감소



#### (3) Pooling

> 사람의 신경망처럼, 고양이의 뇌에 전선을 연결하여 고양이의 신경망이 어떻게 활성화되는지 테스트하는 생물학적 실험을 토대로 만들어짐. 고양이에게 물체를 보여주면, 모든 신경망이 활성화되는 것이 아니라 어떤 객체에 따라서 커다란 특징을 잡아내게 되고, 그 커다란 특징에 따라서 일부 신경망에서만 활성화가 일어났음. 전체적인 특징이 아니라 커다란 특징만 전달해도 물체를 판별하는 데 큰 무리가 없을 것이다(맥스풀링)
>
> 맥스풀링을 쓰면 손실되는 데이터가 나오게 됨
>
> 

왜 그랬을까? 컴퓨터 사양때문

이미지가 축소되어 데이터가 손실되는 것을 감수하더라도 어느정도 성능이 나오기 함

최근에는 많이 사용하지 않는 추세(컴퓨터 성능 향상)

차라리 batch normalization 같은 것을 추가하여 성능을 향상시킴

cnn에 maxpooling이 없는 경우도 많다

기술이 발달하면서 과거의 기술들은 생략되는 경우가 생기게 된다



- 크기: (2, 2)에서 일반적으로 건드리지 않는다: window: (2,2), stride: 2
- Max Pooling, Average Pooling

- 어떤 이미지를 분간할 때, 인간도 사실 모든 픽셀을 보지 않는다. 가장 큰 특징을 봄
- 일반적으로 Maxpooling 사용: 제일 큰 값만 추출
- Average Pooling: 데이터 특징 소실을 우려. 평균값
- 그러나 성능이 가장 좋은 것은 Maxpooling 임이 경험적으로 밝혀짐. 주로 Maxpooling 사용

- 일반적으로 convolutional layer와 maxpooling layer를 섞어서 쓰게 된다

- CNN만으로는 최종적인 분류, 예측문제를 해결할 수 없어서 FC(Fully-connected) DNN Layer를 붙여줘야 한다

  (결국은 일자로 펴준다)



#### (4) Padding

> 이미지에 무언가를 덧대서 갑자기 줄어드는 것을 방지

- 120 * 160 이미지가 처리 후 116 * 156 으로 줄어들지 않도록 이미지를 크게 만들어서 convoluntional 후에 120 * 160 으로 만들어줌



- 출력크기를 조정할 목적으로 사용
- 합성곱 연산을 수행하기 전에 Input_Data 주변을 0으로 채우는 것
- Layer를 깊게 쌓고 싶으면 padding을 준다
- 통상 원본과 같은 크기가 나오도록 구성하므로 "same padding"이라고 한다





### 3) CNN의 구성

#### (1) Channel

- n차원 데이터: n차원 Filter를 사용하여 합성곱 연산 수행
- Input_Data의 채널 수와 Filter의 채널 수는 같아야 함
- 컬러일 경우 이미지가 3개 채널(RGB)을 가지고 있고, 필터도 3개 채널이 있어야 한다
- 채널은 자동으로 맞춰지므로 고민할 필요는 없다



#### (2) Classification

- CNN의 마지막 단계에 분류를 위한 필터를 적용(Sigmoid or Softmax)



#### 일반 이미지를 가지고 왔을 때 문제점

1) jpg마다 픽셀 크기가 다 다르다. 어떻게 할것인가?

- 크기를 맞추는 방법은 있으나, 크기를 맞추면 비율이 달라진다

2) 라벨링 과정이 필요(답을 제공하기 위해)

- 방법1: 디렉토리가 다르면 다른 것으로 labeling 된 것으로 인식
- 방법2: 이미지가 정렬된 대로 labels.txt 파일을 만들어 labeling 해줌 ex) 개구리 트럭 트럭 개구리
- 강아지의 종을 구분한다면 모두 다른 디렉토리에 분류
- 일반적으로 데이터를 수집할 때 디렉토리를 따로 설정하는 방법을 쓴다







- conv1, conv2, conv3 으로 가면 갈수록, 한 픽셀이 보는 이미지의 범위는 점점 넓어지게 된다

  (이미지의 정보를 함축하여 보게 됨)

  - "내부적인 표현을 함축적으로 학습한다"고 표현
  - "상위개념으로 추상화된다"고 표현

- 오토인코더의 "표현학습" 개념이 후에 등장





노란책 p.191

fit_generator:  tf v1 문법, fit으로 통일됨



p.218

model.evaluate_generator: test_generator



### 4) Overfitting Issues

더 많은 데이터를 사용하는 것 같은 기법!



#### Image Augmentation(이미지 증강)

> 노란책 p.193
>
> 빨간책 p.134, 노란책 p.312
>
> SMOTE와 목적이 비슷(적은 데이터를 증폭시킴)

- Overfitting을 회피하기 위해 더 많은 데이터를 생성하여 사용
- 원본이미지를 여러가지 다른 형태로 만듦









- 케라스 함수형 API: 함수형모델로 auto encoder 구현하는 것 보게될것 (함수로 객체를 만들어감)







### 5) 전이학습(Transfer Learning)

> Learning한 것을 전달받음
>
> 이미 학습된 파라미터를 전달받겠다

- 학습되었다는 것은? 우리가 넣은 데이터의 특징을 확인할 수 있다는 의미
- 1000가지 이미지를 분류하도록 모델을 학습시켜두었다면, 1000가지 이미지를 분류하는 파라미터(필터) 들은 그 중 일부인 강아지와 고양이를 처리할 수도 있을 것이다



#### ImageNet Large Scale Visual Challenge

- 1000가지의 이미지 클래스를 분류하는 문제

- 2012년에 딥러닝 CNN을 활용(AlexNet) 하면서 Error rate 이 10퍼센트 이상 떨어짐
- 인간의 에러율 5% 정도. 사람보다 더 정확한 모델들이 생겨남
- 이미 학습된 것을 전달받아 쓰겠다는 것

- 학습된 파라미터의 집합: 모델
- 사전학습된 Parameter(Model)를 가져와서 적용하자!



##### 고려해야 할 사항

1. Input shape을 맞춰주면 됨

2. DNN(Classifcation Layer) 재사용이 불가능할 가능성이 큼

   ex) 원본은 1000개 분류인데, 나는 2개만 혹은 10개만



- 굉장히 많은 모델들이 이미 학습되어있다



#### CNN의 종류(중요한 모델들)

##### (1) LeNet

- 1) 얀 르쿤(Yann Lecun)연구팀에서 개발한 최초의 CNN 구조
  - 1) 인공지능 4대천왕 중 한명
- 합성곱(Convolution), Subsampling(Max Pooling), Full Connection(Classification)
- 

##### (2) AlexNet

- 2012 ILSVC 우승(Alex Khrizevsky)
- LeNet과 유사한 구조이며, 2개의 GPU로 병렬연산 수행



##### (3) GooLeNet(InceptionNet)

- 2014 ILSVC 우승(Google)
- 가로 세로 모두 convolution 과 pooling을 깊이 쌓음
- Inception module: 총 9개의 가로방향으로 깊은 구조(inception... 더 깊이.....)



##### (4) ResNet

- Res: Residual 
- 맥스풀링 없이 convolutional layer만 쌓음
- 2015 ILSVC 대회 우승(Microsoft 북경연구소)
- GoogleNet(22 layers)의 약 7배인 152 layers로 구성
- Layer가 깊어져 학습이 안되는 문제(경사하강)를 Skip Connection을 통해 해결



##### (5) VGGNet

- 2014 ILSVC 대회 준우승(University of Oxford - Visual Geometry Group)
- layer의 개수에 따라 VGG16과 VGG19두가지 모델로 구성





#### Feature Extraction

- 사전에 학습된 모델을 사용하여 특성 추출(Feature Extraction)
  - Feature Extraction: Parameter 재사용
  - Classification: Parameter학습



#### Fine Tuning

- 시작하는 레이어에서는 가까운 것만 학습하지만, convolutional이 일어날수록 더 많은 영역을 보게 된다

- VGG는 1000개를 구분하지만, 앞쪽은 비슷비슷할 것
- 뒤로 갈수록 특징이 구체적으로 발현된다

- 처음, 중간은 기존에 학습한 걸 쓰고(공통속성)(freeze), 
- 마지막 레이어를 우리 데이터에 맞게 재학습시킴(Unfreeze)

- 일반적으로 feature extraction하는 것보다 더 잘된다



#### Visualizing Intermediate Activations

> 노란책 p. 225

- CNN으로 우리가 보는 것은 Image Processing (다른 분야에 써도 상관없지만)
- Transfer
  1. Layer 구조만 가져와서 파라미터는 새로 학습
     - 시간이 오래걸림
  2. Fine Tuning 방식: 학습된 파라미터의 일부만 재학습
  3. Feature Extraction: 파라미터를 그대로 가져와서 사용
     - Layer를 통과시켜 특징 맵만 뽑아냄
     - 잘 쓰이지는 않음



프로젝트 시) 다양한 것을 시도하다가, transfer learning 을 선택하게 되었다. 는 스토리텔링이 필요할 것



- 자연어: 이미지처럼 간단하게 transfer 할 수 없다
- 갖다쓰는 게 안된다!





강화학습의 기본 철학: 데이터 없이 학습한다

- 데이터를 수집하면서 학습한다
- 가상세계에서 데이터를 생성해가면서 학습한다
- CPS가 구성되어야 가능하다
- 이것이 digital transformation의 끝판왕...





### 사전훈련된 모델 다루기

> 빨간책 p.241



- 텐서플로 허브





만약 움직이는 이미지라면? 정확도와 속도 사이에서 고민해야할 것





### 6) 추가적인 학습 

#### (1) Object Detection



##### R-CNN : Region CNN

##### YOLO(You Only Look Once)

- Bounding Box Coordination(조정)
  - Grid 별로 분류를 하게 됨
  - Grid별로 같은 객체라고 판단하면, 확장시켜나가는 형태로 학습함
  - YOLO(CNN)을 통해 나온 긴 모양의 output을  prediction tensor라고 한다
  - prediction tensor에 세가지 속성이 들어간다
    - 1: Bounding Box의 정보
    - 2: Bounding Box에 대한 신뢰정보(0~1 사이), 객체가 아닌 것은 모두 background
    - 3: 클래스 분류 확률값
- YOLO는 prediction tensor를 사용해서 한번만에 이것을 처리함







- 객체인식의 평가기준 (Object Detection Metric) 2가지
  1. mAP(mean Average Precision): 정확도
  2. FPS(Frame Per Seconds): 신속성(속도)



#### (2) Image Segmentation

> 빨간책 p.378, 386

- 이미지의 픽셀단위로 구분하는 것
- 픽셀 단위로 구분하도록 학습 ex) 픽셀1은 하늘, 픽셀 200은 산, ... 
- 

#### (3) Image Captioning

- Image detection + natural language processing(LSTM)





## 7. Recurrent Neural Network

> - 시계열(time series)
> - Recurrent: 순환/반복
> - 지금까지는 Layer 단계에서 forward만 했지, 순환하는 것은 없었다
> - 데이터만의 특징이 중요했음
> - CNN: 누적될 수는 있겠지만, 앞의 내용이 뒤에 영향을 주지는 않는다. "기억이 전달되지 않는다"



- 순차적으로 input이 들어오고, 데이터들 간 시간의 순서에 따른 관계가 있을 것이다

  ex) 한 글자를 치면 다음 글자를 예측

- 들어오는 순서에 따른 특징을 가지게 될 것이다! 

- 시간순서, 앞뒤 순서에 따라 데이터를 처리



- sketch_rnn_demo
- 사진을 rnn으로 처리할 수도 있지만, 성능이 좋아지기는 어렵다
- 그러나 영상이라면 달라질 수 있다 (앞에서 뭘 했는지 알아야 뒤에서 뭘 할지 예측하기 쉬우므로)
- 언어도 비슷한 원리 (최근은 자연어처리에 rnn을 쓰지는 않는다)



- 앞쪽의 특징이 뒤에도 영향을 주는 데이터



#### Feed-forward Neural Network와의 차이점

- 1) 전 단계의 기억(Short-Term Memory)을 가지고 동작
  - 1) 이전 단계의 output
  - 직전 단계의 기억만 들어가는 것

- ex)

  - Input: [[1] , [2] , [3] , [4] , [5]]
  - node에는 activation을 tanh를 한다 (-1 ~ 1 사이): output (-1 ~ 1 사이값)으로 나감
  - [2]가 들어갈 때, [1]이 들어와서 같이 처리가 됨
    -  [1]
    - 첫번째 단계에서 처리된 결과 + [2]
    - 두번째 단계에서 처리된 결과 + [3] 
  - hidden layer가 하나지만, 5개인 것처럼 느껴짐
  - input 사이즈에 따라 반복하는 것처럼 보임

  - hidden layer를 여러 개 만들 수도 있다. 쌓는다고 한다 (stacked RNN) 



- 심플하게 만들어져있음 (그래서 문제가 생김)





#### DNN과 CNN신경망

- hidden layer 가 2개 이상

- 각 Layer간의 상태를 기억하지 않고 입력과 출력이 독립적으로 처리
- 각 Layer마다 독립적으로 가중치(Weight)를 학습



#### 순환신경망: 내부에 루프가 존재하는 신경망

- 구조상으로는 hidden layer가 하나밖에 없는 deep learning 모델
  - hidden layer가 하나면 mlp, 여러개면 deep learning

- 은닉층(Hidden Layer)의 출력이 계속 순환하면서 입력값과 함께 학습에 사용
- 연속적(Sequence) 데이터 처리를 위해서는 이전 단계의 정보가 필요
- 학습 단계에서 1) 모든 Layer가 같은 가중치를 **공유**
  - 1) 실제 Hidden Layer는 1개, 같은 파라미터를 공유해서 사용함
  - 실제로 w이 한개... 계속 누적해서 학습시킴
  - 파라미터의 개수는 적으나 학습시간이 짧지는 않다

- 순환신경망의 형태
  - one to one: 이미지 분류
  - one to many: 이미지 설명 문장 (이미지 -> "강아지" 가 "풀밭" 에서 "뛰고" 있어요)
  - many to one: 여러개의 입력에 대한 감성 분석(긍정, 부정)
    - "영화"가 "굉장히" "재미있어요" -> 긍정
    - "영화"가 "엄청" "지루해요" -> 부정
    - 단어가 몇개 들어올지는 모르지만
  - many to may: 기계번역(영어문장 -> 한글문장)
    - 단어의 개수나 길이가 달라질 수 있다



---

#### 기억해둘 것

RNN은 Short-Term Memory를 가지고 동작하는데, 이것은 바로 직전 단계의 기억을 가지고 기억한다는 의미이다!

(후에 문제가 됨)

Long-term momery가 필요한 문제들도 있는데, short-term memory만 사용

이것을 해결하기 위해 만든 것이 LSTM(Long Short-Term Memory)이다 

---



- 순차적인 정보를 처리하기 위한 모델

  - 앞뒤순서(상호관계)가 존재하는 시계열 데이터

  - 텍스트나 음성 데이터 처리(번역, 음성인식, 음악, 동영상)

    - 최근에는 많이 사용하지 않는 추세

    

- ht <- tanh(Wx * Xt + Wh * ht-1 + b)

  - single layer가 반복
  - Sequence Data 처리에 적합

- Issues

  - Long-Term Dependency: 입력이 깊어지면(문장이 길어지면) 처음 들어온 것의 기억이 소멸됨
  - BPTT(Back-Propagation Through Time)
    - time: 입력의 길이
    - time이 길어지면, 많이 반복하는 거라 layer가 깊어지는 현상 발생
    - hidden layer가 깊어지는 현상으로 **vanishing gradient** 현상이 발생하게 됨



- LSTM이 나오게 된 이유



- sequence-to-sequence 모델도 나타나게 된다





## 8. Long Short-Term Memoy

> 장기, 단기를 모두 가지고 있다
>
> - Long-Term Memory: 장기기억
> - Short-Term Memory: 단기기억



- 수식이 상당히 복잡하다
- 이것을 개선하기 위해 GRU가 등장했지만 심플하다보니 성능이 좋지않아서, LSTM이 더 많이 쓰인다



- 기존 RNN에 Long-Term Memory(Memory Cell) 구조 추가
  - Long-Term Dependency Issue 해결
  - Vanishing Gradient 및 Exploding Gradient Issue(발산) 해결



#### Gate 구조

> 수식이 복잡해 보이지만, W, b는 모두 파라미터
>
> 이것들을 지우고 생각해보면, Xt(=input)과 ht-1(=단기기억)만 남는다
>
> 단기기억을 누적시켜서 장기기억 형태로 만들어 나간다
>
> 복잡해 보이지만 sigmoid(0 ~ 1)와 tanh(-1 ~ 1)만 있는 형태



1. Input 게이트
2. Output 게이트
3. **Forget 게이트**
   - 기억을 잊게 만드는 게이트



- RNN에서는 tanh만 썼지만, LSTM은 sigmoid 3개, tanh 2개로 구성



- 직전단계 Output: O(t-1)
- LSTM은 O(t-1)뿐만 아니라 계속 연결되는 무언가가 하나 더 있다 (Long-Term Memory, **Memory Cell**)
- 곱해주기 전 모두 sigmoid 연산을 한다 (0~1 사이 값을 곱해준다): 기존에 가지고 있던 정보를 몇 퍼센트 정도 남길 것인가?를 의미(비율)



- y_hat과 y의 오차가 줄어드는 데 얼마나 기여하는지(중요한 정보인지)는 tanh로 연산함





#### tanh() vs. sigmoid()

- tanh(): -1 ~ 1
  - 정보의 **강약** 정도(sigmoid 보다 tanh가 성능이 좋으므로 선택한 것)
  - 다음 단계에서 **얼마나 중요한가**를 조정
- sigmoid(): 0 ~ 1
  - 정보의 반영 비율
  - 다음 단계에 **얼마나 반영할지**를 조정 (얼마나 남기고, 얼마나 버릴까(forget)를 결정)





### 1) 게이트의 구조

- C: 장기기억(직전단계에서 넘어오지만 output은 아니다), Memory Cell

- ht-1: 직전단계의 output이자 단기기억

- Xt: input

  

#### (1) Forget Gate

> input이 dogs, 장기기억이 cats 라고 가정

- 새로운 정보(Dogs)를 사용하여 과거의 정보(Cats)를 잊는(sigmoid) 기능
- ht-1 도 고양이 이야기일 가능성이 큼. 갑자기 강아지 이야기가 들어옴 (화제가 전환됨)
- Xt와 ht-1을 더해서 sigmoid 연산함
- 기존에 있던 정보(cats)에 sigmoid한 값을 곱해서 기존에 갖고 있던 내용을 삭제(forget)함
- sigmoid로 어느정도 반영할지를 결정



#### (2) Input Gate

- 잊허진 과거의 정보(Memory Cell)에 새로운 정보(Dogs)를 추가하는 기능

- 새롭게 들어온 내용을 다시 장기기억에 더해주는 역할
- 새롭게 들어온 내용과 기존의 내용을 더해서 sigmoid, tanh 각각 연산
- sigmoid의 결과인 i와 tanh의 결과인 g를 곱해서 장기기억에 새로운 기억을 더해줌



#### (3) Output Gate

- 두가지가 나감
- ht
- Ct
- 장기기억의 내용이 얼마나 중요한 정보인지를 반영한 것과 현재 들어온 기억의 반영여부를 곱하여 다음번 output을 만들어냄

- 단기기억과 장기기억이 적절히 관리되면서 sequence가 길어져도 long-term dependency같은 문제가 발생하지 않음



---

- 모든 과정동안 파라미터들이 역전파에 의해 학습되면서 y_hat이 y와 비슷해지도록 함

---





#### GRU(Gated Recurrent Unit)

- LSTM Memory Cell이 간소화된 버전

- foget gate와 input gate를 중첩시켜 간소화한 버전

- 속도는 빨라지지만 성능이 떨어져 실무에서 쓰진 않는다



#### 시계열 데이터 분석

- 주기성(Periodicity), 추세(Trend), 계절성(Seasonality) 존재

- 시계열 데이터는 시간 순서에 따라 Train/Test를 분리

  ex) Train Dataset: 2011년 01월 01일 ~ 2017년 12월 31일

  ​	  Test Dataset: 2018년 01월 01일 ~ 2019년 12월 31일

- 시계열 데이터 학습 방법

  - 일정기간의 x 데이터로 y를 예측하도록 학습
  - x 데이터: 180일 간의 평균온도
  - y 데이터: 181일 째의 평균온도
  - 며칠로 할지를 정하는 것이 하이퍼 파라미터



- LSTM로 시계열 예측이 생각보다 어렵다
- 전통적인 ARIMA 방식을 쓰는 것이 더 성능이 좋을 수도 있음



---

#### 책 정리

빨간책 p.174, 181~183

노란책 p.264, 268

필요하다면 DNN없이 구현할 수 있다

p.272 LSTM 참조

---



#### 양방향 RNN

노란책 p.297

- 패턴이 있다면, 순방향이든 양방향이든 패턴이 나타남(파라미터가 두배)
- 앞에서 학습, 뒤에서 학습 내용을 합침





## 9. Generative Adversarial Network

> - GAN의 컨셉을 이해하면 자연어처리, 추천시스템에 대해 더 이해할 수 있다
>
> - 지금까지는 지도학습의 관점에서, y를 판별해내는 모델은 만들었다면, Generative Model의 목적은 y와 상당히 유사한 무언가를 생성해내겠다는 것
>
> - 지금까지 한 것과는 굉장히 다른 일!
>
> - "생성적 모델은 창조의 영역이다"고 함 (모방에 가까움)
>
> - 기존처럼 명확하게 학습되는 것이 아니라 학습시키기 난해한 면 존재





AE

VAE 

를 거쳐 GAN으로 이어진다





### 1) AutoEncoder



#### 지금까지의 딥러닝 구조

- 딥러닝 레이어를 구현할 때, input들이 들어오고, 그 사이에 hidden layer를 배열, 마지막 레이어에 output
- 사이에 있는 파라미터들을 학습(업데이트)시키는 것이 목적
- 학습된 파라미터를 가지고 모델로 사용
- CNN, LSTM, DNN이든 마지막 output를 판별하는 것이 지금까지의 모델



#### AutoEncoder

- Encoder: 원래 데이터를 다른 형태로 바꿔주는 것

- Encoder의 반대개념: Decoder

- 인코딩된 정보로부터 원래 정보가 디코딩되어야 함(input X가 들어갔다가 X가 다시 나오도록)

- 인코딩하면서 정보를 압축(축소), 축소된 정보로부터 다시 원래값으로 돌아오도록 만듦

- 인코더: input을 축소된 값으로 바꿔나감

- 디코더: 축소된 값으로부터 원래값을 복원해나감

- 학습을 시킬 때는, 인코더와 디코더를 함께 학습시킴

- 학습이 끝난 후에는 인코더 따로, 디코더 따로 사용 가능 (세 개의 모델처럼 동작)

  1) 인코더+디코더

  2) 인코더

  3) 디코더



- AutoEncoder 목적: Latent Space를 학습하는것

  - 잠재공간(Latent Space): 관측 데이터를 잘 설명할 수 있는 공간(Manifold)
  - Latent Space는 그냥 벡터일 뿐, 우리가 의미를 부여할 수는 있지만 그것이 명확하지 않다(정확히 무엇인지 알 수 없다)

- 차원축소(Dimentionality Reduction): 관찰 데이터 기반의 잠재 공간을 파악하는 것

  ex) 28 * 28 이미지라면, 784(28 * 28)개의 특징을 알고 있어야 이미지를 표현해낼 수 있다

  - AutoEncoder는 숫자 3개(표현벡터)로 줄이고, 이 표현벡터로 숫자 784개를 생성해낼 수 있어야 한다
  - 표현벡터 예) 글자의 크기, 글자의 구부러진 정도, 두께 등 

  - 784개의 특징을 알아야 표현이 가능한 것을 3개의 특징만으로 복원
  - 3개의 숫자로 만들어내는 인코딩 파라미터, 디코딩 파라미터들이 존재

  - 물론 특징이 더 들어가면 더 잘 복원해낼 수 있게 된다



- 일반적으로 원본의 크기보다 작게 학습시키지만, 원본이 그대로 나오도록 학습시킨다
- 모델에서 y가 없다
- y가 없기 때문에 비지도 학습이라고 하지만, 엄밀히 말하면 비지도학습이 아니라 X가 y의 역할을 함



- 인코더와 디코더 두 부분으로 구성

  - 인코더: 인지 네트워크(Recognition Network)로 입력을 내부 표현으로 변환

  - 디코더: 생성 네트워크(Generative Network)로 내부표현을 출력으로 변환

    - 생성보다 엄밀히 보면 복원에 가깝다

  - 입력데이터(x)와 출력데이터(x)가 같음

  - 은닉층의 개수를 줄여 제약을 줌

    - 입력 데이터의 축소, 압축 효과
    - 주성분분석(PCA)으로 구현 가능

    

- 인코더, 디코더를 따로 사용할 수 있어야하기 때문에 단순한 sequential 모델로 만들기 어렵다

- 함수형 API를 사용하는 이유

- 왜 자기자신을 재생성할까? 여러기능(노이즈 제거 가능)







미술관책 3장. 오토인코더 p.87

빨간책

p.327

p.337 이미지 자체를 확대 (Conv2DTranspose)

p.343 클러스터링 (K-means clustering): 비슷한 숫자들이 비슷한 공간에 모여있을까?







비지도학습

### t-SNE(Stohchastic Manifold Embedding)

- t-분포를 사용
- 각 데이터의 유사도를 확률적(Stochastic)으로 표현(밀도함수로 봄)
  - 거리가 멀면 확률이 떨어지고, 거리가 가까우면 확률이 올라갈 것이다
- 하나의 데이터로부터 다른 데이터에 대한 거리를 't-분포'의 확률로 치환
  - 멀리 떨어질수록 likelihood값이 떨어짐
  - 주변의 몇개의 데이터까지 볼 것인지를 hyperparameter로 지정
  - 가까운 거리의 데이터는 높은 확률값
  - 먼 거리의 데이터는 낮은 확률값
- 고차원과 저차원에서 확률값을 계산 후, 저차원 확률값이 고차원에 가까워지도록 학습
- 연산에 많은 시간이 걸리기 때문에 50차원 이하의 데이터 사용을 권장



- 행렬분해: 추천시스템에서 배우게 될것





### 2) Variational AutoEncoder(변이형 오토인코더)

> 일반적으로 "VAE"라고 한다
>
> Variational: "변분추론"
>
> AutoEncoder와의 차이점
>
> - AutoEncoder는 생성모델이 아님(복원)
> - Variational AutoEncoder는 생성모델 (Generative Model)
> - autoencoder는 encoder를 학습시키는 것이 주 목적
> - VAE는 decoder(=generator, 생성기)를 학습시키는 것이 목적
>
> AutoEncoder이긴 한데, AutoEncoder처럼 동작하진 않는다
>
> 목적이 정 반대지만 생긴 것이 비슷하게 생겼다
>
> AutoEncoder: 앞쪽에 원본데이터를 개념벡터로 차원을 축소하고, 원본데이터로 복원시키는 것이 목적
>
> VAE: input X와 output X는 다르다(새롭게 만들어진 X)
>
> 새롭게 만들어진 output X가 원본데이터와 매우 흡사하게 만드는 것



- AE: latent space의 값을 확정된 값 하나로 포인팅함

- VAE: 1) 잠재공간의 값을 가우시안 확률분포(정규분포) 값의 범위로 제공

  - 1) 잠재공간의 값 = 개념벡터
  - 정규분포의 모수는 평균/분산만 알면 모양을 알 수 있다
  - 실제 잠재공간의 값을 우리가 가진 데이터의 평균과 분산으로 학습시키는 것

  - 평균과 분산을 알면, 그와 조금 벗어난 다른 데이터를 추론해낼 수 있다는 논리
  - 평균은 AE의 latent space 값과 같고, 분산은 정규분포 값과 같도록 값을 제공함

  

- 변이형 오토인코더는 latent space의 크기가 정해져있음 (평균, 분산)
- AE는 인코더로 학습하고 나면, decoder로 같은 사진을 복원함
- VAE는 분포를 학습시키고 random으로 분산을 뽑아내면(평균은 바꾸지 않는다), 특징이 살짝 바뀐 비슷한 새로운 그림이 **생성** 된다



- VAE는 입력이미지가 통계적 과정을 통하여 생성되었다고 가정

  - 이미지를 2개의 벡터 1) z_mean과 2) z_log_var로 매핑

    - 1) 평균의 개념벡터
    - 2) 분산의 개념벡터

  - 이 벡터는 잠재 공간상 확률분포를 정의하고 1) 디코딩을 위한 포인트 샘플링에 사용됨

    - 1)  랜덤한 포인트 샘플링
    - 똑같은 것을 가지고 학습하면 복원일 뿐
    - 약간 다른 포인트를 샘플링하여 원본처럼 생긴 전혀 다른 이미지를 생성해냄
    - 기존 것이 아니라 비슷하게 생긴 다른 것을 만들어내고 싶은 것!
    - 상당히 유사하지만 동일하지 않은 무언가가 만들어지게 됨

    

- VAE를 발전시키면, 모바일 앱(눈을 키워주는, 얼굴을 줄여주는)

  - 눈, 얼굴을 처리하는 개념벡터를 찾아서 분포를 이동하여 새로운 이미지 생성





---

#### 정리

AE와 VAE는 모양만 같다

AE는 자기가 축소시킨 데이터를 원복하는 것이 목적

복원하는 포인트를 찾으려고 노력한다

앞쪽의 encoder를 통해 latent space를 찾는 것이 목적



반면에 VAE는

latent space를 고정된 값을 학습시키는 것이 아니라, 정규분포의 확률로 학습

따라서 평균과 분산을 찾아냄

찾아낸 값을 사용해서, 임의의 값을 샘플링하여 비슷하지만 새로운 값을 만들어내도록 학습하는 것



오토인코더는 인코더를 학습하는 것이 목적이지만,

VAE는 디코더를 학습하는 것이 목적 (평균, 분산 데이터를 가져와야 하므로 인코더가 필요한 것)

---





- 잠재공간의 정규분포에서 z값을 무작위 샘플링하여 1) 디코더로 복원

  - 1) 디코더 = 생성기(generator)

- VAE는 구조적이고 연속적인 1) 잠재공간(=개념벡터) 표현을 생성

  - 1) 벡터이므로 연산이 가능함: 개념벡터의 의미를 알고 있다면, 벡터에서 더하기 빼기를 하면서 이미지를 변형할 수 있다(우울한사진 ~ 미소)
  - VAE를 통해서 개념공간 자체를 정규분포 형태로 표현해낸다면, 학습시킨 것에서 약간 변하는 형태가 될 것

  

- 자연어처리에도 나타남 ex) 왕자 - 남자 + 여자 = 공주

  - 문장 간의 비교가 가능해짐
  - 단어를 embedding 하는 모양이 autoencoder와 똑같이 생김

- 개념벡터를 만들어낼 수 있으면 이미지, 텍스트, 음악, 목소리 등을 만들어내는 것이 가능해짐



- 잠재공간의 경로를 따라 한 숫자가 다른 숫자로 자연스럽게 변경

  노란책 p.382, 386, 392 (잠재공간의 경로에 따라서 특징이 바뀌어가며 나타남) 





미술관책 p.105 AE와 VAE에 있는 인코더의 차이점





---

원본이미지에 학습된 값으로 랜덤샘플링하여 원본이미지와 유사한 새로운 이미지를 만들어내겠다!

---





VAE가 발전한 형태가 GAN이다!





### 3) Generative Adversarial Network(GAN)

>  원본이미지의 특징을 보는 게 없이, 원본이미지를 만들어낼 수 있을까?





- 두 개의 sub모델(인코더, 디코더)이 적대적인 방식으로 학습한다



- 상반된 목적의 두 신경망 모델이 경쟁을 통하여 학습하고 결과물을 생성
- 판별자(Discriminator)도 굉장히 중요해지고 있다 ex) 딥페이크



- VAE는, 원본이미지의 분포에서 샘플링된 변형된 데이터가 들어와서 유사한 이미지를 생성하는 것이므로 얼토당토않은 데이터가 들어가지 않음
- GAN은 garbage값(noise)이 input으로 들어감, 이미지가 생성됨
- 판별자: 진짜인지 가짜인지만 분류하는 이진분류 모델(특별하지 않다)
- 문제는 Generator... fake이미지를 잘 만들어야 진짜인지 가짜인지 판별자가 헷갈림
- Generator가 이미지를 잘 만들 수 있도록 (판별자를 속일 수 있도록) 학습하는 것
- generator의 목적: 판별자를 속이는 것
- discriminator의 목적: generator에게 속지 않는 것



- generator가 학습이 잘 되었다는 뜻: 판별자가 진짜인지 가짜인지 구분을 못함
- discrimintor의 정확도가 떨어짐
- 서로 반복해가면서 학습을 시킴(Generator가 학습할 때 discriminator는 학습하지 않음)







- 생성자(Generator) vs. 판별자(Discriminator)
- 생성자는 랜덤데이터를 기반으로 실제 데이터 같은 거짓데이터를 생성
  - 실제에 가까운 거짓 데이터를 생성하는 것이 목적
  - 판별자를 속이지 못한 데이터를 입력받아 반복학습
- **판별자**는 생성자가 만든 데이터가 **실제인지 거짓인지 판별**하도록 학습
  - 생성자의 **거짓 데이터에 속지 않는 것**이 목적
  - 생성자에게 속은 데이터를 입력받아 반복학습



- 학습시키는 것이 굉장히 어렵다







### Style Transfer

(빨간책 설명이 자세하므로 참조)

빨간책 p. 294, p.325  예제 - 눈에 확 들어오지 않음

노란책 p. 372, p.381





## 10. Natural Language Processing(NLP)



- 학습된 모델로 번역, 문장 요약, 문장 생성 등의 작업 수행
- 대량의 말뭉치(Corpus)를 모델 학습에 활용
  - 자연어 연구를 위하여 **특정 목적**을 가지고 수집한 언어의 표본
    - 목적에 맞는 말뭉치를 수집해야 한다
  - 다국어 처리를 위해 여러 나라 말을 넣어 처리하는 경우도 있다
- 자연어 처리의 목적은 이해(Understanding)가 아님
  - 사람은 이야기한 것을 기억하고 있다. 기계는 기억하고 있을까?
    - ex) 어제 내가 이야기한 거 있잖아
  - 기억에 기반한 이야기들을 어떻게 리턴할 것인가?
  - 연산(Computation)이나 처리(Processing)의 영역
  - 오차를 줄이거나, 유사도를 높이는 방식
- 수학적 연산을 위하여 언어를 **숫자로 변환**하는 작업이 필요
- 인간이 사용하는 자연어를 컴퓨터가 연산할 수 있는 1) **벡터(Vector)**로 변환
  - 1) Vector: 숫자의 나열





### 1) Preprocessing

- 자연어 학습에 적합하도록 수집된 텍스트를 사전 처리작업이 필요(Cleansing)



- Tokenization

  - 일정한 단위로 잘라서, 필요없는 단어 빼고, 원형으로 바꾸는 작업

  - Sentence(문장단위), word(글자 단위), Character(자음/모음/알파벳 단위)
  - Stop Words(불용어): 특별히 의미없는 것(조사, 반복어, 관사 등)
  - Stemming, Lemmatization: 시제, 변화하는 것(복수, 과거분사 등)을 원형으로 바꿔주는 것


- Encoding


  - 정수인코딩
  - 원-핫 인코딩



#### Tokenization

- 수집된 말뭉치(Corpus)를 토큰(Token)단위로 나누는 작업
  - 토큰의 단위는 일반적으로 의미를 가지는 단위로 정의
- 일반적으로 형태소(Morpheme) 단위의 토큰화 수행
  - 형태소란 의미를 가지는 가장 작은 말의 단위를 의미
- Tokenization을 "형태소 분석기"라고도 한다



- 단어 토큰화(Word Tokenization): 단어를 기준으로 나눈 것
- 문장 토큰화(Sentence Tokenization): 토큰의 단위가 문장인 경우





### 2) Language Model

- 언어(단어, 문장)에 존재하는 현상을 표현하기 위해 확률을 할당하는 것

  - 확률적 모델링

- 문장(Sequence)이 적절한지, 말이 되는지 판단하기 위한 기준

  - P(승객이 버스에 탔다) vs. P(승객이 버스에 태운다)

    - 어느 것이 더 자연스러운가? 1번이 훨씬 자연스럽다
    - 확률적으로 1번이 더 높다

  - 나는 딥러닝을 P(배운다) vs. P(어렵다) vs. P(고친다) vs. P(가르친다)

    - 어느 것이 가장 확률이 높은가? 그리고 그것을 어떻게 학습시킬 것인가?

    

- BoW(Bag of Words): 개선되어 n-gram, TF-IDF 가 개발됨

- n-gram

- TF-IDF(Term Frequency - Inverse Document Frequency)





#### (1) Bag of Words

> 백에 단어들을 모두 집어넣는 개념
>
> Bag: 단어들의 집합



- 문서가 가지는 모든 단어(Words)를 **문맥이나 순서를 무시하고** 일괄적으로 문장에 포함된 단어에 대해 **빈도값**을 부여해 특징을 추출

- 발생 빈도가 높을수록 **중요한 단어**로 인식

  - 자주 등장하는 단어가 중요한 단어일 것이다

  

- 장점

  - 쉽고 빠른 구축
  - 예상보다 문서의 특징을 잘 반영 (생각보다 잘 동작함)

- 단점

  - 언어의 특성상 자주 등장하는 단어에 높은 중요도를 부여
    - stop words를 제외하더라도, 자주 등장 = 중요
  - 단어가 순서를 고려하지 않아, 문맥 의미(Semantic Context) 반영 부족(거의 되지 않음)
  - **희소 행렬(Sparse Matrix)**을 생성하여 학습 시간 증가 및 성능에 부정적 영향



- Bow - Feature Vectorization

  - M개의 문서(Docoment) 또는 문장(Sentence)
  - 모든 1) 단어(Term) 추출 시 N 종류의 단어 존재
    - 1) Term = word
  - M * N 크기의 행렬(Term-Document Matrix) 생성
    - 문서크기 * 단어종류

  

ex) A: I like dog, B: You like dog, C: I hate bug

  각 문장이 3개의 단어

|      | I    | like | dog  | You  | hate | bug  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| A    | 1    | 1    | 1    |      |      |      |
| B    |      | 1    | 1    | 1    |      |      |
| C    | 1    |      |      |      | 1    | 1    |

  - 3 * 6 Matrix가 만들어진다

  - 값은 있다/없다를 의미하는 것이 아니라 빈도수를 의미한다

  - A의 (1,1,1,0,0,0)을 벡터처럼 본다

  - A-B, B-C 중 어느 것이 더 유사한가? 

    - 사람이 봐도 A-B
    - 컴퓨터의 연산법: 곱해서 더함 (A-B의 경우 1\*1 + 1*1 = 2, B-C는 0)
    - 사람과 비슷하게 동작한다

    

- 우리가 1차원적인 문장만 만들지는 않는다

- 문제점 모든 단어를 다 쪼개놓음

- 순서상 의미가 있는 단어들(숙어), 문맥이 고려되지 않음





#### (2) n-gram

> Tokenization을 하는데, 최소단위로 쪼개는 것이 아니라 여러 개로 쪼개봄
>
> 단어를 연속적으로 보려고 하는 것
>
> n: 하이퍼 파라미터, 1-gram 2-gram 3-gram... 어느 게 성능이 더 좋을까?



- BoW의 단어 순서를 무시하는 단점을 보완

  - 1-gram(Unigram), 2-gram(Bigram), 3-gram(Trigram)

    ex) Machine Learning is fun and is not boring

    - Machine과 Learning을 분리해서는 안된다

    

- Sparse matrix는 더 희소해짐 (Matrix가 더 커짐)

  - 커지는 만큼 표현할 수 있는 것은 많아지게 됨



- 여전히 빈도수가 높은 단어에 중요도를 부여함
- 그런데 만약 그냥 많이 나오는 단어라면?





#### (3) TF-IDF(Term Frequency - Inverse Document Frequency)

> BoW는 단어의 빈도수만 중요하다고 생각
>
> 그런데 많이 나올 수 밖에 없는 구조라면?
>
> 단어의 빈도수도 고려하지만, 문서 전체에서 자주 나오는 단어라면 중요도를 감쇠함
>
> "단순히 빈도로만 보지 않겠다"



- BoW의 단어의 빈도수만 고려하는 단점을 보완
  - 개별 문서(문장)에 자주 나타나는 단어에 높은 가중치 부여
  - 모든 문서(문장)에 전반적으로 자주 나타나는 단어에는 **패널티** 부여
- TF Score
  - 특정문서(Document)에 등장한 특정 단어(Term)의 등장 횟수
- IDF Score
  - 특정 단어(Term)가 등장한 문서(Document)의 수



ex)

A: a new book, a used book, a book store (총 9개 단어)

B: a dog in a dog house is big(총 9개 단어)



- TF-Score

  - TF Score: A

  | a    | New  | book | used | store |
  | ---- | ---- | ---- | ---- | ----- |
  | 3/9  | 1/9  | 3/9  | 1/9  | 1/9   |

  - TF Score: B

  | a    | dog  | in   | house | is   | big  |
  | ---- | ---- | ---- | ----- | ---- | ---- |
  | 2/8  | 2/8  | 1/8  | 1/8   | 1/8  | 1/8  |



- IDF Score

  - log(전체 문서의 수 / 해당단어를 포함하는 문서의 수)

  | a            | new            | book           | dog            | big            |
  | ------------ | -------------- | -------------- | -------------- | -------------- |
  | log(2/2) = 0 | log(2/1) = 0.3 | log(2/1) = 0.3 | log(2/1) = 0.3 | log(2/1) = 0.3 |

  - 갭이 클수록 값이 커짐

  

- TF-IDF Score

  - TF-IDF(T) = TF(T) * log(M/IDF(T))
  - 자주 등장하는 단어는 0이 되어버림
  - 특성을 가진 단어들은 값을 가지고 살아나게 됨

  |      | a         | new            | book          | dog            | big            |
  | ---- | --------- | -------------- | ------------- | -------------- | -------------- |
  | A    | 3/9*0 = 0 | 1/9*0.3 = 0.03 | 3/9*0.3 = 0.1 | 0              | 0              |
  | B    | 2/8*0 = 0 | 0              | 0             | 2/8*0.3 = 0.08 | 1/8*0.3 = 0.03 |



- 단순히 빈도수만 고려하는 형태로 동작하지 않음!





---

현재 쓰이고 있는 것은 아니지만,

embedding이 어떻게 등장하게 되었는지 이해하기 위해 배경으로 알아두자!

---



- 정말 유사한가? 

- 빈도수만으로 유사함을 판단

  ex) 대통령이 시민들에게 연설을 했습니다 vs. 문재인이 군중에게 말했다

  - 동일한 단어가 없기 때문에 유사도가 0이다
  - 문재인-대통령, 시민-군중 이 같은 단어라는 것을 알려줘야 한다

  - 그래서 corpus가 중요하다

  

- 단어 자체를 벡터로 변환시켜서 벡터 기반의 단어 유사도를 비교





### 3) Similarity

- 단어나 문장 간 유사도(Similarity)를 비교
  - 단어나 문장을 **벡터로 변환** 후 유사도를 비교

- Euclidean Distance: 거리 기반의 유사도 측정
- Cosine: 벡터의 사잇각 기반의 유사도 측정



#### Euclidean Distance Similarity

- 벡터 간의 거리를 계산하여 유사도를 측정
- ed1 = np.sqrt((5-5)^2 + (1-2)^2)
- ed1: 남자와 왕자의 유사도
- ed2: 남자와 공주의 유사도
- 거리가 짧을수록 유사도 높음



#### Cosine Similarity

- 벡터 간의 사잇각을 계산하여 유사도를 측정
  - 사잇각이 작을수록 유사도 높음
  - 벡터의 크기는 무시
  - 벡터의 크기가 아닌 **방향성** 기반





- Topic까지 학습시키고 싶으면 어떻게 할까를 고민한 것이 Embedding

- 빈도수 기반이 아닌, 다른 특징으로 word를 vector로 표현해보자!





### 4) Embedding

> 종류가 많다 (빨간책 p.203)
>
> Word2Vec, GloVe, FastText, ELMo, BERT, GPT-3 ...



#### Word2Vec

- One-Hot Encoding 및 이전 Similarity 방식의 단점 보완

- 문장 내의 비슷한 위치(neigbor words)에 있는 단어로부터 유사도 획득

  - 이 단어가 어느 단어와 같이 사용되는가?

  

- 각각의 단어 벡터가 **단어 간 유사도를 반영한 값**을 가지고 있음

- 분산표현(Distributed Representation)의 분포가설에 의해 동작

  - 분포가설: 비슷한 위치에 등장하는 단어는 비슷한 의미를 가짐



- 어떻게 학습하는가? 두가지 방법



#### (1) CBOW

- **주변**에 있는 단어를 사용하여 **중간**에 있는 단어를 예측하는 방법
- 2 windows: "The fat **cat** sat on the mat"
  - cat: Center Word
  - Context Word
  - 윈도우 숫자: 하이퍼 파라미터
- 윈도우를 이동하며 생성된 데이터로 임베딩 학습



#### (2) Skip-gram

> 일반적으로 CBOW보다 Skip-gram방식이 성능이 좋다

- 중간에 있는 단어를 사용하여 주변에 있는 단어를 예측하는 방법
- 2 windoes: "The fat **cat** sat on the mat"
- 단어를 Tokenization해서 Vectorization (원핫 인코딩)



- 신경망으로 학습

- Word가 X, neighbor가 y (모두 원-핫 인코딩)
- embedding layer를 학습하게 함
- 우리가 제공한 말뭉치에 따라서 표현(embedding)에 학습이 이루어지게 됨

- embedding layer가 latent space와 같다
- 문서에서의 표현방식을 이렇게 학습시킨다
- 저차원의 밀집벡터가 된다









노란책 p.252

from keras.layers import Embedding

- embedding이라는 layer를 끼워주면, 원하는 차원의 embedding layer를 만들어준다







#### BERT vs. GPT2

- 최근 NLP동향: Transfer Learning 과 Fine Tuning



#### BERT

- 다른 단어와의 관계(Attention)를 통하여 임베딩 매트릭스 생성
- Transformer self-Attention
- 



BERT: Sesame School 의 캐릭터 이름



한국어 말뭉치를 넣어서 파라미터 조정은 다시 해주어야 한다

KoBERT, KoGPT 등 (어떤 말뭉치를 사용하느냐에 따라)



전이학습 알고리즘: BERT, GPT3 활용해보자!



한국어에 적합한 모델을 만들려고 노력중 (뽀로로?)





자연어 평가지표: SSA (Sensibleness and Specificity Average)







노란책 p.254

사전훈련된 단어 임베딩 사용하기

- 이미 임베딩 된 단어를 사용하면 어떨까?



- GloVe는 행렬분해 기법 사용: 하나의 행렬을 두개의 행렬의 곱으로 분해하는 기법





p.258-259





빨간책 p.199~202 임베딩 설명







## 11. Recommendation System

>추천과 관련된 이전 개념
>
>- 연관 규칙(Association Rules): 조건부확률과 발생빈도를 고려
>
>- K-means Clustering
>
>- Embedding: 상품에 대한 embedding (상품에 대한 리뷰)
>
>
>
>- 소비자(User)가 있고, 아이템(상품, 영화, 수업 등...)이 있을 때, 어떤 방식으로 추천해줄 수 있을것인가?



- 사용자 본인도 잘 몰랐던 취향이나 관심사를 추천시스템이 발견하여 제시



#### 관련데이터

1. 사용자가 어떤 상품을 구매했는가? (RDB에 존재)
2. 사용자가 어떤 상품을 둘러보았는가? (시스템 로그파일, 반정형데이터)

- 하둡, 스파크 등을 통해 처리하는 것이 일반적

3. 사용자가 어떤 상품을 장바구니에 넣었는가?

- RDB에 들어있을 가능성이 큼

4. 사용자가 상품에 어떤 평점을 부여했는가? (리뷰, 평점)

- 디테일하게 쓰고 활용하고 싶겠지만, 그러면 사람들이 리뷰를 작성하지 않는다
- 리뷰쓰면 포인트 지급하는 이유: 데이터임

5. 사용자가 가입 시 선택한 관심분야는 무엇인가?

- 추천이 가장 어려운 경우: 신규유입자
- 아무런 데이터가 없다. "콜드 스타트 이슈"
- 콜드 스타트가 추천에서 가장 어려운 문제 중 하나
- 해결하기 위한 방법 중 하나가 신규가입자의 정보 얻기



---

데이터 없이 할 수 있는 것이 없다

---



### 1) 협업 필터링(Collaborative Filtering)

- 사용자의 행동양식(User Behavior) 데이터 (구매, 시청, 클릭 등)
  - 사용자(User)가 아이템(Item)에 매긴 **평점정보**나 상품구매이력
- 1) 최근접이웃(Nearest Neighbor) 방식과 2) 잠재요인(Latent Factor) 방식
  - 1) 거리가 아니라 유사도 상의 이웃을 의미
  - 2) 상품의 구매나 사용에 대한 특징을 표현해내는 잠재공간을 학습
  - 두 방식 모두 '사용자-아이템 행렬' 사용
- **User, Item, 평가점수 세가지는 반드시 있어야 한다**



#### (1) 사용자-아이템 행렬(User-Item Matrix)

- 축적된 '사용자 행동양식(User Behavior)' 데이터 기반

- 사용자가 아직 평가하지 않은 아이템을 '**예측평가**(Predicted Rating)'
  ex) User1이 Item4에 대한 평점이 없으면, 비슷한 User2, User3의 Item4에 대한 평점을 기반으로 예측

  ​	높은 점수가 나오면 추천해주는 방식

|       | Item1 | Item2 | Item3 | Item4 |
| ----- | ----- | ----- | ----- | ----- |
| User1 | 4     |       | 3     | PR    |
| User2 | 5     | 3     |       | 4     |
| User3 |       | 1     | 2     | 3     |



### 2) 최근접 이웃 협업 필터링(Memory 협업 필터링)

- 사용자 기반과 아이템 기반으로 분류

- 사용자 기반과 아이템 기반이 사람의 관점에서는 같아 보일수도 있겠지만, 매트릭스 계산하는 방법이 다르다



#### (1) 사용자 기반 협업필터링(User Based Filtering)



##### 당신과 비슷한 고객들이 다음과 같은 상품도 구매했습니다

- 사용자와 유사한 다른 사용자를 Top-N으로 선정
- Top-N 사용자가 좋아하는 아이템을 추천하는 방식
  1. 특정 사용자와 다른 사용자 간의 '유사도(Similarity)'를 측정
  2. 가장 유사도가 높은 Top-N사용자를 추출
  3. 그들이 선호하는 아이템을 사용자에게 추천



##### User-Item Matrix

- User1과 User2의 'item'에 대한 평점 유사도가 높음
- User1이 아직 구매하지 않은 Item4와 Item5를 추천(Predicted Rating)

|       | Item1 | Item2 | Item3 | Item4 | Item5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| User1 | 5     | 4     | 4     | PR    | PR    |
| User2 | 5     | 3     | 4     | 5     | 3     |
| User3 | 4     | 3     | 3     | 2     | 5     |



#### (2) 아이템 기반 협업필터링

> 일반적으로 아이템 기반 협업필터링의 성능이 더 좋다



##### 이 상품을 구매한 다른 고객들은 다음과 같은 상품도 구매했습니다

- 사용자의 아이템에 대한 **평가결과**가 유사한 아이템을 추천

- 행과 열이 사용자 기반 필터링과 반대

  1. 아이템에 대한 평점 간 '유사도'를 측정

  2. 평점 유사도가 높은 아이템을 추출

  3. 그들이 선호하는 아이템을 사용자에게 추천

     

##### Item-User Matrix

- Item1과 Item2의 평점 유사도가 높음
- User4가 아직 구매하지 않은 Item2를 추천(Predicted Rating)

|       | User1 | User2 | User3 | User4 | User5 |
| ----- | ----- | ----- | ----- | ----- | ----- |
| Item1 | 5     | 4     | 5     | 5     | 5     |
| Item2 | 5     | 4     | 4     | PR    | 5     |
| Item3 | 4     | 2     | 2     |       | 4     |









### 3) 잠재요인 협업 필터링

> 넷플릭스 경진대회에서 우승하며 유명해짐



- User-Item Matrix를 사용하여 '**잠재요인**(Latent Factor)'을 추출
  - '잠재요인'이 무엇인지 명확하게 정의할 수 없음
  - User-Item Matrix를 '잠재요인' 기반의 저차원 밀집행렬인 '사용자-잠재요인'과 '잠재요인-아이템'으로 행렬분해(Matrix Factorization)
    - SVD(Singular Vector Decomposition)
    - NMF(Non-Negative Matrix Factorization)
  - 이 두 행렬의 내적을 통해서 새로운 '사용자-아이템 평점' 행렬 생성
  - 사용자가 평점을 부여하지 않은 아이템에 대한 **'예측평점'** 생성 가능
- '잠재요인'기반의 저차원 밀집행렬인 '사용자-잠재요인'과 '잠재요인-아이템'으로 행렬분해 (차원축소)



#### (1) User-Item Matrix

- 희소행렬(Sparse Matrix)

  ex) 행렬 (User * Item): 4 * 4

|      | Item1 | Item2 | Item3 | ...  |
| ---- | ----- | ----- | ----- | ---- |
| U1   | R1    |       | R1    |      |
| U2   |       | R2    | R2    |      |
| U3   | R3    |       | R3    |      |
| U4   |       | R4    |       |      |

- Rating은 있을수도, 없을수도 있다

- 이 행렬을 두개 행렬의 곱으로 쪼개겠다: (파이썬이므로 @으로 표현)

- 행렬을 두개로 쪼갰을 때, User-latent factor matrix / latent-factor-item matrix

- 확률적 경사하강 기반 행렬분해

  - P와 Q 행렬로 계산된 예측 R 행렬의 값이 실제 R 행렬의 값과 가장 최소의 오류를 가질 수 있도록 반복적으로 'Cost Function' 최적화를 통해 P와 Q를 유추해내는 것

  1. 임의의 값을 갖는 P와 Q 행렬 초기화
  2. P와 Q 행렬로 예측 R 행렬 계산 후 실제 R 행렬과 오차값 계산
  3. 오차를 최소화하도록 P와 Q 행렬의 값을 업데이트
  4. 2단계와 3단계를 반복하며 P와 Q 행렬값을 학습
  5. 행렬 분해가 완료된 P와 Q 행렬로 예측평점 행렬 생성



#### (2) User-'Latent Factor' Matrix

- 4 * 2 (user * latent factor) 생성

|      | LF1  | LF2  |
| ---- | ---- | ---- |
| U1   |      |      |
| U2   |      |      |
| U3   |      |      |
| U4   |      |      |
| ...  |      |      |

- Latent Factor의 차원은 지정해줄 수 있다
- 가격, 브랜드, 장르 등으로 추정



#### (3) 'Latent Factor'-Item Matrix

- 2 * 4

|      | Item1 | Item2 | LF3  | LF4  |
| ---- | ----- | ----- | ---- | ---- |
| LF1  |       |       |      |      |
| LF2  |       |       |      |      |



#### (4) '예측평점' User-Item Matrix

- 밀집행렬(Dense Matrix)

|      | Item1 | Item2 | Item3 | Item4 | Item5 |
| ---- | ----- | ----- | ----- | ----- | ----- |
| U1   | 5     | 4     | 4     | 5     | 5     |
| U2   | 5     | 3     | 4     | 5     | 3     |
| U3   | 4     | 3     | 3     | 2     | 5     |
| U4   | 3     | 2     | 5     | 2     | 4     |

- 밀집행렬 두개로 나뉘어졌다가, 두개를 곱하면 값들이 모두 채워진 밀집행렬이 됨
- 채워진 값을 예측평점이라고 봄
- 이것을 바탕으로 추천





---

#### 정리

- 수학에서는 값이 비어있으면 행렬분해가 되지 않는다
- 따라서 머신러닝적으로 비어진 값을 예측하여 행렬분해한다
- "확률적 경사하강 기반" 행렬분해 라고 한다 (계산식을 고민하지 말자!)
- 이 행렬만 만들 수 있다면 sparse의 값을 채울 수 있고, 아이템 기반의 추천을 할 수 있게 된다



행렬을 인수분해할 수 있고,

인수분해를 확률적 경사하강을 통해 해낸 다음에, 

이를 통해 만들어진 예측평점 행렬을 통해 추천을 해주는 방식

---



#### 행렬분해 기반의 비지도학습(NMF)

머신러닝 책 참고





---

하나의 알고리즘만 쓰지는 않고, 여러가지를 섞어서 쓴다

추천은 비지도학습의 영역이라고 볼 수도 있다

왜? 정답이 없으므로

맞는지 아닌지 알 수 없다. 사람들이 그렇게 움직여야 맞는 것이므로

---



