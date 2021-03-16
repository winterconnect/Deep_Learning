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



- Perceptron, 퍼셉트론이란?
  - Artificial Neural Network(ANN, 인공신경망)의 한 종류
- 신경망은 생물학적인 것



- Multi-Layer란?
  - Layer가 여러개
  - 지금까지 우리가 쓴 것은... single-layer였을까?



신경망도, ML의 지도학습 알고리즘 중 하나 (input, output 필요)

함수형 모델이 발전된 형태 (GD를 이용해 w, b 학습)

발전하여 Deep Learning으로 작동





### Neural Network

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



### Perceptron

> 여러가지 종류로 만들어지는 인공신경망 종류 중 하나



- 인공신경망의 한 종류(선형분리기)
- 가장 간단한 형태의 Forward Network
- 1957년 프랑크 로젠블라트(Frank Rosenblatt)에 의해 고안
  - 그 당시 컴퓨터가 없어서 하드웨어로 구현
  - 그런데 왜 소프트웨어 방식을 쓰는가? 싸니까!
- 동작원리
  - 노드의 입력값(Input)과 가중치(Weight)의 곱을 모두 합함
  - 합한 값이 1) 활성화함수의 2) 임계치보다 크면 1, 작으면 0을 출력
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





### Multi-Layer Perceptron

- y = WX + b
- 쌓았더니, XOR 문제가 풀림

- single perceptron으로 해결할 수 없는 복잡한 문제가 풀리게 됨



- 많이 쌓으면 당연히 capacity 가 좋아진다 (parameter가 많아지므로)



- 다층퍼셉트론: 퍼셉트론으로 해결할 수 없는 비선형 분리문제에 필요

  - Hidden Layer가 하나인 신경모델을 다층 퍼셉트론 이라고 한다

- 여러 층의 퍼셉트론(Node)을 쌓아서 동작(Input Layer, Hidden Layer, Output)

  - Hidden Layer가 몇개인 게 좋은가? 해봐야 안다(Hyperparameter)
  - 늘어날수록 파라미터의 개수가 늘어난다

  



#### Linear Model

- 지금까지는 하나의 모델을 잘 만들려고 노력한 것



#### Nonlinear model

- 여러개의 모델을 쌓아 훨씬 복잡한 문제를 해결하려고 함

- 선 하나하나가 파라미터
- 정확히는 함수가 "노드"



- Hidden Layer의 노드를 늘릴 수는 있지만, Hidden Layer의 깊이를 늘려 하이퍼파라미터를 경사하강으로 학습시키는 것은 불가능하다고 생각했다
- 컴퓨터 기술의 발달로 Hidden Layer를 늘리는 것이 가능해짐





### Deep Neural Network(DNN)

