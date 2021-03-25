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
- 





강화학습의 기본 철학: 데이터 없이 학습한다

- 데이터를 수집하면서 학습한다
- 가상세계에서 데이터를 생성해가면서 학습한다
- CPS가 구성되어야 가능하다
- 이것이 digital transformation의 끝판왕...
- 





### 사전훈련된 모델 다루기

> 빨간책 p.241



- 텐서플로 허브





만약 움직이는 이미지라면? 정확도와 속도 사이에서 고민해야할 것





### 6) 추가적인 학습 

#### (1) Object Detection

- R-CNN : Region CNN
- YOLO



#### (2) Image Segmentation



#### (3) Image Captioning

- Image detection + natural language processing(LSTM)






## 7. Recurrent Neural Network



## 8. Long Short-Term Memory



## 9. Generative Adversarial Network

> GAN의 컨셉을 이해하면 자연어처리, 추천시스템에 대해 더 이해할 수 있다





AE

VAE 

를 거쳐 GAN으로 이어진다



