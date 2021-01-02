from collections import Counter
from typing import List
from LinearArgebra.vector import sum_of_squares
import matplotlib.pyplot as plt

num_friends = [100.0, 49, 41, 40, 25, 21, 21, 19, 19, 18, 18, 16, 15, 15, 15, 15, 14, 14, 13, 13, 13, 13, 12, 12, 11,
               10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
               9, 9, 9, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6,
               6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
               4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
               3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
               1, 1, 1, 1, 1, 1, 1, 1, 1]

# 큰 사이즈의 데이터셋을 잘 설명하기 위해 친구수를 히스토그램으로 나타내기
friends_count = Counter(num_friends)

xs = range(101)  # 최댓값은 100
ys = [friends_count[x] for x in xs]  # 히스토그램의 높이는 해당 친구 수를 가지고 있는 사용자의 수

plt.bar(xs, ys)
plt.axis([0, 101, 0, 25])
plt.title("Histogram of Friend Counts")
plt.xlabel("# of friends")
plt.ylabel("# of people")
plt.show()

# 위의 히스토그램 데이터를 잘 설명하기 위해 간단한 통계를 구해보기

# 데이터 포인트의 갯수
num_point = len(num_friends)
assert num_point == 204

# 최대값과 최소값
largest_value = max(num_friends)
smallest_value = min(num_friends)

assert largest_value == 100
assert smallest_value == 1

# 정렬된 리스트의 특정 위치에 있는 값을 구하기
sorted_values = sorted(num_friends)
second_smallest_value = sorted_values[1]
second_largest_value = sorted_values[-2]

assert second_smallest_value == 1
assert second_largest_value == 49

"""
====================================================================
                       1. 중심 경향성 (central tendency)
====================================================================
"""


# 중심 경향성 (central tendency) 지표는 데이터의 중심이 어디 있는지를 나타내는 지표이다.
# 대부분의 경우 평균(average = mean) 를 사용하게 된다.
def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


assert 7.3333 < mean(num_friends) < 7.3334

"""
중앙값 (median) 구하기

데이터 포인트의 개수가 홀수라면 중앙값은 전체 데이터에서 가장 중앙에 있는 데이터 포인트를 의미한다. 

반면, 데이터 포인트의 개수가 짝수라면 중앙값은 전체 데이터에서 가장 중앙에 있는 두 데이터 포인트의 평균을 의미한다.

평균과 달리 중앙값은 데이터 포인트 모든 값의 영향을 받지 않는다.

예를 들어 값이 가장 큰 데이터 포인트의 값이 더 커져도 중앙값은 변하지 않는다.
"""


def _median_odd(xs: List[float]) -> float:
    """ len(xs) 가 홀수면 중앙값을 반환한다. """
    return sorted(xs)[len(xs) // 2]


def _median_even(xs: List[float]) -> float:
    """ len(xs) 가 짝수면 두 중앙값의 평균을 반환 """
    sorted_xs = sorted(xs)
    hi_midpoint = len(xs) // 2
    return (sorted_xs[hi_midpoint - 1] + sorted_xs[hi_midpoint]) / 2


def median(v: List[float]) -> float:
    """ v의 중앙값을 계산 """
    return _median_even(v) if len(v) % 2 == 0 else _median_odd(v)


assert median([1, 10, 2, 9, 5]) == 5
assert median([1, 9, 2, 10]) == (2 + 9) / 2

assert median(num_friends) == 6

"""
평균은 중앙값보다 계산하기 간편하며 데이터가 바뀌어도 값의 변화가 더 부드럽다.
하지만 평균은 이상치(outlier)에 매우 민감하다.

가령, 위의 예제에서 친구가 가장 많은 사용자가 200명의 친구를 가지고 있다고 해보자.

이런 경우, 평균은 7.82 만큼 증가하겠지만 중앙값은 변하지 않을 것이다.

이상치가 '나쁜' 데이터(이해하려는 현상을 제대로 나타내고 있지 않은 데이터) 라면 평균은 데이터에 대한
잘못된 정보를 줄 수 있다.

예시를 하나 보면, 1980 년대 노스캐롤라이나대학교의 전공 중에서 지리학과 졸업생의 초봉이 가장 높게 조사되었다.
그 이유는 지리학을 전공한 NBA 최고의 스타 마이클 조던의 초봉 때문이었다.
"""

""" 
분위(quantile) 는 중앙값을 포괄하는 개념인데, 특정 백분위보다 낮은 분위에 속하는 데이터를 의미한다.
(중앙값은 상위 50%의 데이터보다 작은 값을 의미한다.)
"""


# 분위 구하기
def quantile(xs: List[float], p: float) -> float:
    """ x의 p 분위에 속하는 값을 반환 """
    p_index = int(p * len(xs))
    return sorted(xs)[p_index]


assert quantile(num_friends, 0.10) == 1
assert quantile(num_friends, 0.25) == 3
assert quantile(num_friends, 0.75) == 9
assert quantile(num_friends, 0.90) == 13


#  최빈값 (mode, 데이터에서 가장 자주 나오는 값) 을 살펴보는 경우도 있다.
def mode(x: List[float]) -> List[float]:
    """ 최빈값이 하나보다 많을수도 있으니 결과를 리스트로 반환 """
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts.items()
            if count == max_count]


"""
====================================================================
                       2. 산포도 (dispersion)
====================================================================
"""


#  산포도 (dispersion) 는 데이터가 얼마나 퍼져 있는지를 나타낸다.
#  보통 0과 근접한 값이면 데이터가 거의 퍼져 있지 않다는 의미이고 큰 값이면 매우 퍼져 있다는 것을 의미하는 통계이다.
# 가장 큰 값과 가장 작은 값의 차이를 나타내는 "범위" 는 산포도를 나타내는 가장 간단한 통계치이다.
def data_range(xs: List[float]) -> float:
    return max(xs) - min(xs)


assert data_range(num_friends) == 99


# 분산 (variance) 은 산포도를 측정하는 약간 더 복잡한 개념이다.
# 편차
def de_mean(xs: List[float]) -> List[float]:
    """ x의 모든 데이터 포인트에서 평균을 뺌 (평균을 0으로 만들기 위해서) """
    x_bar = mean(xs)
    return [x - x_bar for x in xs]


# 분산
def variance(xs: List[float]) -> float:
    """ 편차의 제곱의 (거의) 평균 """
    assert len(xs) >= 2, "variance requires at least two elements"

    n = len(xs)
    deviations = de_mean(xs)
    return sum_of_squares(deviations) / (n - 1)


assert 81.54 < variance(num_friends) < 81.55

"""
위의 식을 살펴보면 편차의 제곱의 평균을 계산하는데, n 대신에 n-1 로 나누는 것을 확인할 수 있다.
이는 편차의 제곱 합을 n으로 나누면 편향(bias) 때문에 모분산에 대한 추정값이 실제 모분산보다 작게 계산되는 것을 보정하기 위해서이다.
https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation
"""
import math


# 분산의 단위는 기존 단위의 제곱이기 때문에, 분산대신 원래 단위와 같은 단위를 가지는 표준편차(standard deviation) 을 사용할 때가 많다.
def standard_deviation(xs: List[float]) -> float:
    """ 표준편차는 분산의 제곱근 """
    return math.sqrt(variance(xs))


assert 9.02 < standard_deviation(num_friends) < 9.04

"""
범위와 표준편차 또한 평균처럼 이상치에 민감하게 반응하는 문제가 있다.
더 안정적인 방법은 상위 25% 에 해당되는 값과 하위 25% 에 해당되는 값의 차이를 계산하는 것이다.
"""


def inter_quartile_range(xs: List[float]) -> float:
    """ 상위 25%에 해당되는 값과 하위 25%에 해당되는 값의 차이를 반환 """
    return quantile(xs, 0.75) - quantile(xs, 0.25)


assert inter_quartile_range(num_friends) == 6

"""
====================================================================
                       3. 상관관계
====================================================================
"""

from LinearArgebra.vector import dot

"""
"사용자가 사이트에서 보내는 시간과 사용자의 친구 수 사이에 연관성이 있다" 라는 가설을 검증해보자.
"""
# 사용자가 하루에 데이텀을 몇분동안 하는가? 에 대한 데이터 리스트
daily_minutes = [1, 68.77, 51.25, 52.08, 38.36, 44.54, 57.13, 51.4, 41.42, 31.22, 34.76, 54.01, 38.79, 47.59, 49.1,
                 27.66, 41.03, 36.73, 48.65, 28.12, 46.62, 35.57, 32.98, 35, 26.07, 23.77, 39.73, 40.57, 31.65, 31.21,
                 36.32, 20.45, 21.93, 26.02, 27.34, 23.49, 46.94, 30.5, 33.8, 24.23, 21.4, 27.94, 32.24, 40.57, 25.07,
                 19.42, 22.39, 18.42, 46.96, 23.72, 26.41, 26.97, 36.76, 40.32, 35.02, 29.47, 30.2, 31, 38.11, 38.18,
                 36.31, 21.03, 30.86, 36.07, 28.66, 29.08, 37.28, 15.28, 24.17, 22.31, 30.17, 25.53, 19.85, 35.37, 44.6,
                 17.23, 13.47, 26.33, 35.02, 32.09, 24.81, 19.33, 28.77, 24.26, 31.98, 25.73, 24.86, 16.28, 34.51,
                 15.23, 39.72, 40.8, 26.06, 35.76, 34.76, 16.13, 44.04, 18.03, 19.65, 32.62, 35.59, 39.43, 14.18, 35.24,
                 40.13, 41.82, 35.45, 36.07, 43.67, 24.61, 20.9, 21.9, 18.79, 27.61, 27.21, 26.61, 29.77, 20.59, 27.53,
                 13.82, 33.2, 25, 33.1, 36.65, 18.63, 14.87, 22.2, 36.81, 25.53, 24.62, 26.25, 18.21, 28.08, 19.42,
                 29.79, 32.8, 35.99, 28.32, 27.79, 35.88, 29.06, 36.28, 14.1, 36.63, 37.49, 26.9, 18.58, 38.48, 24.48,
                 18.95, 33.55, 14.24, 29.04, 32.51, 25.63, 22.22, 19, 32.73, 15.16, 13.9, 27.2, 32.01, 29.27, 33, 13.74,
                 20.42, 27.32, 18.23, 35.35, 28.48, 9.08, 24.62, 20.12, 35.26, 19.92, 31.02, 16.49, 12.16, 30.7, 31.22,
                 34.65, 13.13, 27.51, 33.2, 31.57, 14.1, 33.42, 17.44, 10.12, 24.42, 9.82, 23.39, 30.93, 15.03, 21.67,
                 31.09, 33.29, 22.61, 26.89, 23.48, 8.38, 27.81, 32.35, 23.84]

daily_hours = [dm / 60 for dm in daily_minutes]


# 우선 분산과 비슷한 개념인 공분산(covariance) 부터 살펴보자. 분산은 하나의 변수가 평균에서 얼마나 멀리 떨어져 있는지 계산한다면,
# 공분산은 두 변수가 각각의 평균에서 얼마나 멀리 떨어져 있는지 살펴본다.
def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs and ys must have same number of elements"

    return dot(de_mean(xs), de_mean(ys)) / (len(xs) - 1)


assert 22.42 < covariance(num_friends, daily_minutes) < 22.43
assert 22.42 / 60 < covariance(num_friends, daily_hours) < 22.43 / 60

"""
하지만 공분산을 해석하는 것은 다음과 같은 이유 때문에 쉽지 않다.

1. 공분산의 단위는 입력 변수의 단위들을 곱해서 계싼되기 때문에 이해하기 쉽지 않다.
(예를 들어, 친구 수 * 하루 사용량(분) 이라는 단위는 무엇을 의미하는 것인가?)

2. 만약 모든 사용자의 하루 사용량은 변하지 않고 친구 수만 두 배로 증가한다면 공분산 또한 두 배로 증가할 것이다.
하지만 생각해보면 두 변수의 관계는 변하지 않았다. 다르게 얘기하면, 공분산의 절대적인 값만으로는 '크다' 고 판단하기 어렵다는 것이다.
"""


# 위와 같은 이유 때문에 공분산에서 각각의 표준편차를 나눠 준 상관관계 (correlation) 을 더 자주 살펴본다.

def correlation(xs: List[float], ys: List[float]) -> float:
    """ xs와 ys의 값이 각각의 평균에서 얼마나 멀리 떨여져 있는지 계산 """
    stdev_x = standard_deviation(xs)
    stdev_y = standard_deviation(ys)

    if stdev_x > 0 and stdev_y > 0:  # 두 변수의 공분산을 구해서 각각의 표준편차로 나누어준다.
        return covariance(xs, ys) / stdev_x / stdev_y
    else:
        return 0  # 편차가 존재하지 않는다면 상관관계는 0 이다.


assert 0.24 < correlation(num_friends, daily_minutes)
