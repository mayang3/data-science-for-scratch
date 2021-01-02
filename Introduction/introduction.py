from collections import Counter
from collections import defaultdict

"""
====================================================================
                       1-1. 핵심 인물 찾기
====================================================================
"""

users = [
    {"id": 0, "name": "Hero"},
    {"id": 1, "name": "Dunn"},
    {"id": 2, "name": "Sue"},
    {"id": 3, "name": "Chi"},
    {"id": 4, "name": "Thor"},
    {"id": 5, "name": "Clive"},
    {"id": 6, "name": "Hicks"},
    {"id": 7, "name": "Devin"},
    {"id": 8, "name": "Kate"},
    {"id": 9, "name": "Klein"}
]

friendship_pairs = [(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)]

# 사용자별로 비어 있는 친구 목록 리스트를 지정하여 딕셔너리를 초기화
# result -> {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []}
friendship = {user["id"]: [] for user in users}

# friendship_pairs 내 쌍을 차례대로 살펴보면서 딕셔너리 안에 추가
for i, j in friendship_pairs:
    friendship[i].append(j)
    friendship[j].append(i)


# Q1. 네트워크 상에서 각 사용자의 평균 연결 수는 몇 개인가?
def number_of_friends(user):
    """friends number of user"""
    return len(friendship[user["id"]])


total_connections = sum(number_of_friends(user) for user in users)
avg_connections = total_connections / len(users)

print(total_connections)  # 24
print(avg_connections)  # 2.4

# Q2. 친구가 가장 많은 사람 구해보자.
num_friends_by_id = [(user["id"], number_of_friends(user)) for user in users]

# 친구 수를 기준으로 역순 정렬
num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1], reverse=True)

# found degree centrality (연결 중심성)
# result -> [(1, 3), (2, 3), (3, 3), (5, 3), (8, 3), (0, 2), (4, 2), (6, 2), (7, 2), (9, 1)]
print(num_friends_by_id)

"""
이 지수는 계산하기 쉽다는 장점이 있지만, 항상 기대하는 결과를 가져다 주지는 않는다.
예를 들어, 데이텀 네트워크를 살펴보면 Dunn(id: 1) 은 세 개의 연결 고리를 갖고 있지만 Thor(id: 4) 는 두 개의 연결 고리밖에 갖고 있지 않다.

하지만 네트워크상에서는 Thor(id: 4) 가 가운데에 위치하여 더 중심적인 역할을 하는 것처럼 보인다.
"""

"""
====================================================================
                       1-2. 데이터 과학자 추천하기
====================================================================
"""


# Q1. 친구 추천(친구의 친구를 소개해주는...) 기능을 설계 해주세요.

# 사용자의 친구의 친구 리스트를 불러온다.
def foaf_ids_bad(user):
    """foaf 의 의미는 친구의 친구 (friend of a friend)를 의미하는 약자이다."""
    return [foaf_id
            for friend_id in friendship[user["id"]]  # outer loop
            for foaf_id in friendship[friend_id]]  # inner loop


# 위의 함수를 users[0], 즉 Hero 에 관해 실행했다고 해보자.
# result -> [0, 2, 3, 0, 1, 3]
print(foaf_ids_bad(users[0]))


# Q2. 이번에는 서로 함께 아는 친구 (mutual friends) 가 몇 명인지 세어보자.
def friends_of_friends(user):
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendship[user_id]  # 사용자의 친구 개개인에 대해
        for foaf_id in friendship[friend_id]  # 그들의 친구들을 세어 보고
        if foaf_id != user_id and foaf_id not in friendship[user_id]  # 사용자 자신과 사용자의 친구는 제외
    )


# 이제 Hero(id:0) 은 함께 아는 친구가 Chi(id:3) 과 함께 두명이라는 것을 알 수 있다.
# result -> Counter({3: 2})
print(friends_of_friends(users[0]))

# 직원들의 관심사 데이터 interests 를 손에 넣었다.
# 비슷한 관심사를 가진 사람들을 찾아주는 함수를 만들어 보자.
interests = [
    (0, "Hadoop"), (0, "Big Data"), (0, "HBase"), (0, "Java"), (0, "Spark"), (0, "Storm"), (0, "Cassandra"),
    (1, "NoSQL"), (1, "MongoDB"), (1, "Cassandra"), (1, "HBase"), (1, "Postgres"),
    (2, "Python"), (2, "scikit-learn"), (2, "scipy"), (2, "numpy"), (2, "statsmodels"), (2, "pandas"),
    (3, "R"), (3, "Python"), (3, "statistics"), (3, "regression"), (3, "probability"),
    (4, "machine learning"), (4, "regression"), (4, "decision trees"), (4, "libsvm"),
    (5, "Python"), (5, "R"), (5, "Java"), (5, "C++"), (5, "Haskell"), (5, "Programming languages"),
    (6, "statistics"), (6, "probability"), (6, "mathematics"), (6, "theory"),
    (7, "machine learning"), (7, "scikit-learn"), (7, "Mahout"), (7, "neural networks"),
    (8, "neural networks"), (8, "deep learning"), (8, "Big Data"), (8, "artificial intelligence"),
    (9, "Hadoop"), (9, "Java"), (9, "MapReduce"), (9, "Big Data")
]

# 키가 관심사, 값이 사용자 ID
# default 값이 list 인 dictionary
user_ids_by_interests = defaultdict(list)

for user_id, interest in interests:
    user_ids_by_interests[interest].append(user_id)

# 키가 사용자 ID, 값이 관심사
interests_by_user_id = defaultdict(list)

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)


def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]  # 유저의 관심사를 가져와서
        for interested_user_id in user_ids_by_interests[interest]  # 해당 관심사에 매칭되는 모든 유저들을 가져온다.
        if interested_user_id != user["id"]  # 나는 제외하고 나와 동일한 관심사를 가진 유저들을 리턴해준다.
    )


# 9 번과 관심사가 3개가 겹치고, 1번과 관심사가 2개가 겹치고, 8 번은 1개가 겹치고, 5번은 1개가 겹친다.
# Counter({9: 3, 1: 2, 8: 1, 5: 1})
print(most_common_interests_with(users[0]))

# 익명화된 연봉 데이터
# 각 사용자의 연봉(salary) 이 달러로, 근속 기간(tenure) 이 연 단위로 표기되어 있다.
salaries_and_tenures = [
    (83000, 8.7), (88000, 8.1),
    (48000, 0.7), (76000, 6),
    (69000, 6.5), (76000, 7.5),
    (60000, 2.5), (83000, 10),
    (48000, 1.9), (63000, 4.2),
]

# 근속 연수에 따라 평균 연봉이 어떻게 달라지는지 확인해보자.
# 키는 근속 연수, 값은 해당 근속 연수에 대한 연봉 목록
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# 키는 근속 연수, 값은 해당 근속 연수의 평균 연봉
average_salary_by_tenure = {
    tenure: sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

# 그런데 근속 연수가 같은 사람이 한 명도 없어서 결과가 쓸모가 없다.
# 사용자 개개인의 연봉을 보여 주는 것과 다르지가 않기 때문이다.
# {8.7: 83000.0, 8.1: 88000.0, 0.7: 48000.0, 6: 76000.0, 6.5: 69000.0, 7.5: 76000.0, 2.5: 60000.0, 10: 83000.0, 1.9: 48000.0, 4.2: 63000.0}
print(average_salary_by_tenure)


# 차라리 아래와 같이 경력을 몇개의 구간으로 나누고, 각 연봉을 해당 구간에 대응시켜 보자.
def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"


# 키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 연봉
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# 키는 근속 연수 구간, 값은 해당 구간에 속하는 사용자들의 평균 연봉
average_salary_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}

# {'more than five': 79166.66666666667, 'less than two': 48000.0, 'between two and five': 61500.0}
print(average_salary_by_bucket)
