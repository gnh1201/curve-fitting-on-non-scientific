# curve-fitting-on-non-scientific
Curve fitting on a non-scientific applications - Draft

비과학 어플리케이션에서의 곡선접합(커브피팅) - 초안

## Why?
비과학 어플리케이션(CRM/ERP, 인프라 모니터링 시스템 등)에서 커브피팅(Curve Fitting)을 하는 것은 제한 사항이 많아 까다로운 문제로 알려져 있습니다. 가장 큰 이유로는 과학 및 공학 분야에서 많이 사용되는 소프트웨어 또는 라이브러리로 알려진 matlab, scipy, R 등의 사용이 제한되기 때문입니다.

가령, Python 언어의 인기 라이브러리인 scipy의 경우 커브피팅을 위한 방법으로 `scipy.optimize.curve_fit`이라는 API를 제공하는데, 이러한 이미 알려진 커브피팅 구현체의 혜택을 받지 못하는 것은 많이 안타까운 상황입니다.

본인은 비과학 어플리케이션에 종사하는 업체 최소 3곳에서 커브피팅이 필요한 데이터가 발견될 경우 깊게 분석하지 말라는 상위 실무자의 지시를 받은 경험이 있습니다.

이것은 비과학 어플리케이션 업체들에게서 인기있는 "데이터 분석 솔루션"이라 불리는 상용 소프트웨어들이 산술 평균, 산술 최소, 산술 최대 이 3가지 정보만을 가지고 모든 의사결정을 하도록 구성되어 있다는 문제에서 기인합니다. 나는 이러한 문제를 여러번 경험해보고 보다 더 쉬운 접근법이 없는지 고민했습니다.

나는 커브피팅이 필요한 문제, 특히 사인파와 같은 주기적 데이터가 발견되는 상황에서 비과학 어플리케이션에서 쓰이는 산술적 계산에만 의존하는 데이터 분석 솔루션을 어떻게 지원할 것인지를 고민할 것입니다.

## 예상쟁점
* 추가적인 프로그래밍 언어 지원 및 패키지 설치 없이 바로 지원할 수 있는가? 예를 들어, PHP, Java, Ruby, C# 등으로 작성된 일반(비과학) 어플리케이션에 바로 붙일 수 있는가?
* 사인파의 형태를 나타내는 데이터가 주어졌을 때, 단순한 통계 정보(산술 평균, 최소값, 최대값 등)만을 활용하는 솔루션과 그 솔루션이 제공하는 기능(예: 평균 값에 기초한 이상 탐지, 최소/최대 임계값)에 호환되도록 데이터를 변환할 수 있는가?

## ChatGPT says:
비과학 어플리케이션(CRM/ERP, 인프라 모니터링 시스템 등)에서 커브 피팅을 적용하기 위해서는 다음과 같은 주요 쟁점과 해결책을 고려해야 합니다.

### 예상 쟁점

1. **추가적인 프로그래밍 언어 지원 및 패키지 설치 없이 바로 지원할 수 있는가?**
   - 비과학 어플리케이션은 종종 Python (scipy), MATLAB, R과 같은 과학 라이브러리를 사용할 수 없는 환경에 구축되어 있습니다. 대신 PHP, Java, Ruby, C# 등의 언어로 작성된 경우가 많습니다.

2. **단순한 통계 정보와 호환되는 커브 피팅 솔루션 제공**
   - 주기적 데이터(예: 사인파)가 주어졌을 때, 기존의 단순 통계 정보(산술 평균, 최소값, 최대값 등)만을 활용하는 솔루션과 호환되도록 데이터를 변환할 수 있어야 합니다. 
   - 예를 들어, 평균 값에 기초한 이상 탐지, 최소/최대 임계값 설정 등에 커브 피팅 결과를 적용할 수 있어야 합니다.

### 해결책 및 전략

#### 1. 다양한 프로그래밍 언어로 커브 피팅 구현

각 언어별로 커브 피팅 알고리즘을 구현하여 비과학 어플리케이션에 바로 적용할 수 있도록 합니다.

**Java**:
```java
import org.apache.commons.math3.fitting.CurveFitter;
import org.apache.commons.math3.fitting.WeightedObservedPoint;
import org.apache.commons.math3.fitting.HarmonicFitter;
import org.apache.commons.math3.analysis.function.HarmonicOscillator;

public class CurveFittingExample {
    public static void main(String[] args) {
        CurveFitter<HarmonicOscillator.Parametric> fitter = new HarmonicFitter(new HarmonicOscillator.Parametric());
        fitter.addObservedPoint(new WeightedObservedPoint(1.0, 0.0, 1.0));
        fitter.addObservedPoint(new WeightedObservedPoint(1.0, 1.0, 0.5));
        // 추가 데이터 포인트를 추가합니다.

        double[] parameters = fitter.fit(new HarmonicOscillator.Parametric(), new double[] {1.0, 0.1, 0.0});
        System.out.println("Amplitude: " + parameters[0] + ", Angular Frequency: " + parameters[1] + ", Phase: " + parameters[2]);
    }
}
```

**PHP**:
PHP는 커브 피팅 라이브러리가 부족하지만, PHP에서 C 라이브러리를 호출하거나 Python 스크립트를 실행하는 방식을 사용해 해결할 수 있습니다.

```php
<?php
$command = escapeshellcmd('python curve_fitting_script.py');
$output = shell_exec($command);
echo $output;
?>
```

**Python** 스크립트 예제:
```python
import numpy as np
from scipy.optimize import curve_fit

# 데이터 생성
x_data = np.linspace(0, 4 * np.pi, 50)
y_data = np.sin(x_data) + 0.1 * np.random.normal(size=x_data.size)

# 사인 함수 모델 정의
def sine_function(x, a, b, c):
    return a * np.sin(b * x + c)

# 커브 피팅
popt, _ = curve_fit(sine_function, x_data, y_data, p0=[1, 1, 0])

# 피팅 결과 출력
print(f"Amplitude: {popt[0]}, Angular Frequency: {popt[1]}, Phase: {popt[2]}")
```

**Ruby**:
```ruby
require 'gsl'

x_data = GSL::Vector.linspace(0, 4 * Math::PI, 50)
y_data = x_data.collect { |x| Math.sin(x) + 0.1 * GSL::Rng.alloc(GSL::Rng::DEFAULT).gaussian }

model = GSL::MultiFit::FdfSolver.alloc(GSL::MultiFit::FdfSolver::LEVENBERG_MARQUARDT, x_data.size, 3)
model.set_func { |x, params| params[0] * Math.sin(params[1] * x + params[2]) }
model.set_params([1.0, 1.0, 0.0])

model.iterate

puts "Amplitude: #{model.params[0]}, Angular Frequency: #{model.params[1]}, Phase: #{model.params[2]}"
```

**C#**:
```csharp
using MathNet.Numerics;
using MathNet.Numerics.LinearRegression;
using MathNet.Numerics.LinearAlgebra.Double;

double[] xData = Generate.LinearSpaced(50, 0, 4 * Math.PI);
double[] yData = xData.Select(x => Math.Sin(x) + 0.1 * (new Random().NextDouble() - 0.5)).ToArray();

var model = Fit.Curve(xData, yData, (x, p) => p[0] * Math.Sin(p[1] * x + p[2]), new double[] { 1, 1, 0 });
Console.WriteLine($"Amplitude: {model.Item1}, Angular Frequency: {model.Item2}, Phase: {model.Item3}");
```

#### 2. 데이터 변환 및 호환성 확보

주기적 데이터가 주어졌을 때 이를 단순한 통계 정보로 변환하는 방법을 도입합니다. 예를 들어, 주기적 데이터를 처리하는 경우, 해당 데이터의 평균, 최소값, 최대값을 활용해 단순화된 통계 정보를 제공합니다.

```python
import numpy as np

# 사인파 데이터 생성
x_data = np.linspace(0, 4 * np.pi, 50)
y_data = np.sin(x_data) + 0.1 * np.random.normal(size=x_data.size)

# 평균, 최소, 최대값 계산
mean_value = np.mean(y_data)
min_value = np.min(y_data)
max_value = np.max(y_data)

print(f"Mean: {mean_value}, Min: {min_value}, Max: {max_value}")
```

#### 3. 상용 데이터 분석 솔루션과 통합

기존의 데이터 분석 솔루션이 제공하는 기능과 커브 피팅을 통합할 수 있는 방법을 모색합니다. 예를 들어, 사인파 피팅을 통해 얻은 주기적 패턴을 기반으로 이상 탐지 기능을 향상시킬 수 있습니다.

```python
import numpy as np
from scipy.optimize import curve_fit

# 데이터 생성
x_data = np.linspace(0, 4 * np.pi, 50)
y_data = np.sin(x_data) + 0.1 * np.random.normal(size=x_data.size)

# 사인 함수 모델 정의
def sine_function(x, a, b, c):
    return a * np.sin(b * x + c)

# 커브 피팅
popt, _ = curve_fit(sine_function, x_data, y_data, p0=[1, 1, 0])

# 이상 탐지 (예: 평균에서 일정 범위 벗어나는 값)
fitted_data = sine_function(x_data, *popt)
threshold = np.mean(fitted_data) + 2 * np.std(fitted_data)
outliers = y_data[y_data > threshold]

print(f"Outliers: {outliers}")
```

### 결론

비과학 어플리케이션 환경에서 커브 피팅을 적용하려면, 다양한 언어에 맞춘 커브 피팅 알고리즘을 개발하고, 기존의 단순한 통계 정보 기반 솔루션과 통합할 수 있는 방안을 모색해야 합니다. 이를 통해 주기적 데이터를 효율적으로 분석하고, 데이터 분석의 정확성을 높일 수 있습니다.
