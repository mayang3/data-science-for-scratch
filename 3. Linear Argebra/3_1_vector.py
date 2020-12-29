from typing import List
import math

"""
====================================================================
                       3-1. Vector
====================================================================
"""

"""
간단히 말하면, 벡터(vector) 는 벡터끼리 더하거나 상수(scalar) 와 곱해지면 새로운 벡터를 생성하는 개념적인 도구이다.
더 자세하게는, 벡터는 어떤 유한한 차원의 공간에 존재하는 점들이다. 대부분의 데이터, 특히 숫자로 표현된 데이터는 벡터로 표현할 수 있다.

수많은 사람들의 키, 몸무게, 나이에 대한 데이터가 주어졌다고 해보자. 그렇다면 주어진 데이터를 (키, 몸무게, 나이) 로 구성된 3차원 벡터로 표현할 수 있을 것이다.
또 다른 예로, 시험을 네 번 보는 수업을 가르친다면 각 학생의 성적을 (시험1 점수, 시험2 점수, 시험3 점수, 시험4 점수)로 구성된 4차원 벡터로 표현할 수 있을 것이다.

벡터를 가장 간단하게 표현하는 방법은 여러 숫자의 리스트로 표현하는 것이다.
예를 들어, 3차원 벡터는 세 개의 숫자로 구성된 리스트로 표현할 수 있다.
"""

Vector = List[float]

height_weight_age = [70, 170, 40]  # 인치, 파운드, 나이
grades = [95, 80, 75, 62]  # 시험1 점수, 시험2 점수, 시험3 점수, 시험4 점수


# 1. 벡터 덧셈
def add(v: Vector, w: Vector) -> Vector:
    """ 각 성분끼리 더한다. """
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i + w_i for v_i, w_i in zip(v, w)]


# 2. 벡터 뺄셈
def subtract(v: Vector, w: Vector) -> Vector:
    """ 각 성분끼리 뺀다. """
    assert len(v) == len(w), "vectors must be the same length"

    return [v_i - w_i for v_i, w_i in zip(v, w)]


# 3. 모든 벡터의 합
def vector_sum(vectors: List[Vector]) -> Vector:
    """ 모든 벡터의 각 성분들끼리 더한다. """
    # vectors 가 비어있는지 확인
    assert vectors, "no vectors provided!"

    # 모든 벡터의 길이가 동일한지 확인
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors), "different sizes!"

    # i번째 결과값은 모든 벡터의 i번째 성분을 더한 값
    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]  # outer loop ->
    # 위의 sum 연산에 괄호가 들어가 있으므로 아래 comprehension 이 outer loop 가 된다.


assert vector_sum([[1, 2], [3, 4], [5, 6], [7, 8]]) == [16, 20]


# 4. 벡터의 스칼라곱
def scalar_multiply(c: float, v: Vector) -> Vector:
    """ 모든 성분을 c 로 곱하기 """
    return [c * v_i for v_i in v]


assert scalar_multiply(2, [1, 2, 3]) == [2, 4, 6]


# 5. 벡터의 성분별 평균
def vector_mean(vectors: List[Vector]) -> Vector:
    """ 각 성분별 평균을 계산 """
    n = len(vectors)
    return scalar_multiply(1 / n, vector_sum(vectors))


# 6. 벡터의 내적(dot product) -> 각 성분별 곱한값을 더해준 값
def dot(v: Vector, w: Vector) -> float:
    """ (v_1 * w_1) + ... + (v_n * w_n) """
    assert len(v) == len(w), "vectors must be same length"

    return sum(v_i * w_i for v_i, w_i in zip(v, w))


assert dot([1, 2, 3], [4, 5, 6]) == 32  # 1 * 4 + 2 * 5 + 3 * 6


# 7. 벡터 각 성분의 제곱값의 합
def sum_of_squares(v: Vector) -> float:
    """ (v_1 * w_1) + ... + (v_n * w_n) """
    return dot(v, v)


assert sum_of_squares([1, 2, 3]) == 14  # 1 * 1 + 2 * 2 + 3 * 3


# 8. 벡터의 크기
def magnitude(v: Vector) -> float:
    """ 벡터 v 의 크기를 반환 """
    return math.sqrt(sum_of_squares(v))


# 9. 두 벡터간의 거리
def distance(v: Vector, w: Vector) -> float:
    return magnitude(subtract(v, w))
