# curve-fitting-on-non-scientific
Curve fitting on a non-scientific applications (Draft)

## Why?
비과학 어플리케이션(CRM/ERP, 인프라 모니터링 시스템 등)에서 커브피팅(Curve Fitting)을 하는 것은 제한 사항이 많아 까다로운 문제로 알려져 있습니다. 가장 큰 이유로는 과학 및 공학 분야에서 많이 사용되는 소프트웨어 또는 라이브러리로 알려진 matlab, scipy, R 등의 사용이 제한되기 때문입니다.

Python의 인기 라이브러리인 scipy의 경우 커브피팅을 위한 방법으로 `scipy.optimize.curve_fit`이라는 API를 제공하는데, 이러한 이미 알려진 커브피팅 구현체의 혜택을 받지 못하는 것은 많이 안타까운 상황입니다.

본인은 비과학 어플리케이션에 종사하는 업체 최소 3곳에서 커브피팅이 필요한 데이터가 발견될 경우 깊게 분석하지 말라는 상위 실무자의 지시를 받은 경험이 있습니다.

이것은 비과학 어플리케이션 업체들에게서 인기있는 "데이터 분석 솔루션"이라 불리는 유상 소프트웨어들이 기본적으로 산술 평균, 산술 최소, 산술 최대 이 3가지 정보만을 가지고 모든 의사결정을 하도록 구성되어 있다는 문제에서 기인합니다. 나는 이러한 문제를 여러번 경험해보고 보다 더 쉬운 접근법이 없는지 찾고 있습니다.

나는 커브피팅이 필요한 문제, 특히 사인파와 같은 주기적 데이터가 발견되는 상황에서 비과학 어플리케이션에서 쓰이는 산술적 계산에만 의존하는 데이터 분석 솔루션을 어떻게 지원할 것인지를 고민할 것입니다.

## ChatGPT says:
비과학 어플리케이션에서 커브 피팅을 적용하는 데에는 몇 가지 중요한 도전과제와 해결책이 있습니다. 이 문제에 접근하는 데 있어서 고려해야 할 요소와 전략을 제안해 보겠습니다.

### 도전과제

1. **제한된 라이브러리 사용**: 과학 및 공학 분야에서 일반적으로 사용되는 라이브러리(예: MATLAB, scipy, R 등)의 사용이 제한됩니다.
2. **데이터 분석 솔루션의 한계**: 비과학 어플리케이션에서 사용하는 데이터 분석 솔루션이 주로 단순한 통계 정보(산술 평균, 최소값, 최대값 등)에 의존하고 있습니다.
3. **커브 피팅에 대한 거부감**: 실무자들이 커브 피팅을 사용한 데이터 분석을 꺼리는 경향이 있습니다.

### 해결책

1. **대안 라이브러리 및 도구 탐색**:
   - Python의 `numpy`와 같은 기본 라이브러리 또는 자체 구현한 간단한 커브 피팅 알고리즘을 활용해볼 수 있습니다.
   - 예를 들어, Least Squares 방법을 직접 구현해보는 것도 하나의 방법입니다.

2. **교육 및 인식 개선**:
   - 실무자들에게 커브 피팅의 이점과 필요성을 교육할 필요가 있습니다. 이를 통해 데이터 분석의 정확성과 예측 가능성을 높일 수 있음을 강조할 수 있습니다.
   - 간단한 예제와 시각적 자료를 통해 커브 피팅의 장점을 설명하면 더 효과적일 수 있습니다.

3. **단순한 모델로 시작**:
   - 처음부터 복잡한 모델을 도입하기보다는, 간단한 사인파 피팅과 같은 기본적인 주기 함수 피팅부터 시작해보는 것이 좋습니다.
   - 점진적으로 복잡도를 높여가면서 신뢰를 구축할 수 있습니다.

4. **커스터마이징 가능한 솔루션 개발**:
   - 기존 데이터 분석 솔루션에 간단한 커브 피팅 기능을 추가할 수 있는 플러그인이나 확장 모듈을 개발하는 것도 고려해볼 수 있습니다.
   - 이를 통해 기존 시스템과의 호환성을 유지하면서 기능을 확장할 수 있습니다.

### 구체적인 접근법

1. **사인파 커브 피팅 예제**:
   ```python
   import numpy as np
   from scipy.optimize import curve_fit
   import matplotlib.pyplot as plt

   # 데이터 생성
   x_data = np.linspace(0, 4 * np.pi, 50)
   y_data = np.sin(x_data) + 0.1 * np.random.normal(size=x_data.size)

   # 사인 함수 모델 정의
   def sine_function(x, a, b, c):
       return a * np.sin(b * x + c)

   # 커브 피팅
   popt, _ = curve_fit(sine_function, x_data, y_data, p0=[1, 1, 0])

   # 피팅 결과 시각화
   plt.scatter(x_data, y_data, label='Data')
   plt.plot(x_data, sine_function(x_data, *popt), color='red', label='Fitted curve')
   plt.legend()
   plt.show()
   ```

2. **단계별 접근**:
   - **Step 1**: 간단한 사인파 데이터 생성 및 시각화.
   - **Step 2**: 기본 커브 피팅 알고리즘을 이해시키기 위해 Python 스크립트 제공.
   - **Step 3**: 실무자와 협력하여 실제 데이터에 적용해보고 피드백 반영.

3. **추가적인 지원**:
   - **문서화**: 커브 피팅 방법론과 사용법에 대한 상세한 문서 제공.
   - **워크샵 및 교육 세션**: 실무자를 대상으로 하는 교육 프로그램 진행.

비과학 어플리케이션에서 커브 피팅을 성공적으로 도입하기 위해서는 기술적인 접근 뿐만 아니라, 실무자의 인식 개선과 교육도 함께 병행되어야 합니다. 이를 통해 데이터 분석의 정확성과 효율성을 높이고, 궁극적으로 비즈니스 의사결정에 더 큰 가치를 제공할 수 있을 것입니다.
