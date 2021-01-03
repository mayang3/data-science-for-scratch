from typing import Tuple, List, Callable

"""
====================================================================
                       3-2. Matrix
====================================================================
"""

# 타입 명시를 위한 별칭들
Matrix = List[List[float]]
Vector = List[float]


def shape(A: Matrix) -> Tuple[int, int]:
    """ (열의 개수, 행의 개수) 를 반환 """
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0  # 첫 번째 행의 원소의 개수
    return num_rows, num_cols


assert shape([[1, 2, 3], [4, 5, 6]]) == (2, 3)  # 2행 3열


def get_row(A: Matrix, i: int) -> Vector:
    """ A의 i번째 행을 반환 """
    return A[i]  # A[i]는 i번째 행을 나타낸다.


def get_column(A: Matrix, j: int) -> Vector:
    """ A의 j번째 열을 반환 """
    return [A_i[j] for A_i in A]  # 각 A_i 행에 대해 A_i 행의 j 번째 원소


# 형태가 주어졌을때, 형태에 맞는 행렬을 생성하는 함수
def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    (i, j) 번째 원소가 entry_fn(i, j) 인 num_rows * num_cols 리스트를 반환
    """
    return [[entry_fn(i, j)  # i 가 주어졌을 때, 리스트를 생성한다.
             for j in range(num_cols)]  # [entry_fn(i, 0), ...]
            for i in range(num_rows)]  # 각 i 에 대해 하나의 리스트를 생성한다.


# 위 함수를 이용해서 5 * 5 단위 행렬 만들기
def identity_matrix(n: int) -> Matrix:
    """ n*n 단위 행렬을 반환 """
    return make_matrix(n, n, lambda i, j: 1 if i == j else 0)


assert identity_matrix(5) == [[1, 0, 0, 0, 0],
                              [0, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 1]]

""" 행렬의 쓰임새 3가지  """

# 1. 각 벡터를 행렬의 행으로 나타냄으로써 여러 벡터로 구성된 데이터셋을 행렬로 표현할 수 있다.
# 예를 들어, 1,000명에 대한 키, 몸무게, 나이가 주어졌다면 1,000 * 3 행렬로 표현할 수 있다.

data = [[70, 170, 40],
        [65, 120, 26],
        [77, 250, 19]]

# 2. k 차원의 벡터를 n 차원 벡터로 변환해주는 선형 함수를 n*k 행렬로 표현할 수 있다.

# 3. 행렬로 이진관계 (binary relationship) 을 표현할 수 있다. (예) 그래프에서의 연결관계
