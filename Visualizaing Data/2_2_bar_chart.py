from matplotlib import pyplot as plt
from collections import Counter

"""
====================================================================
                       2-2. 막대 그래프
====================================================================
"""

"""
bar chart(막대 그래프) 는 discrete(이산적인) 항목들에 대한 변화를 보여 줄 때 사용하면 좋다.
아래 예는 여러 영화가 아카데미 시상식에서 상을 각각 몇개 받았는지를 보여준다.
"""

movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

# 막대의 x 좌표는 [0, 1, 2, 3, 4], y 좌표는 [num_oscars] 로 설정
plt.bar(range(len(movies)), num_oscars)

plt.title("My Favorite Movies")  # 제목을 추가
plt.ylabel("# of Academy Awards")  # y축에 레이블을 추가

# x축 각 막대의 중앙에 영화 제목을 레이블로 추가한다.
plt.xticks(range(len(movies)), movies)
plt.show()

"""
====================================================================
                       2-1. 히스토그램
====================================================================
"""
"""
막대 그래프를 사용하면 히스토그램도 그릴 수 있다.
히스토그램이란 정해진 구간에 해당되는 항목의 개수를 보여줌으로써, 값의 분포를 관찰할 수 있는 그래프 형태이다.
"""

grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

# 점수는 10점 단위로 그룹화한다. 100점은 90점대에 속한다.
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],  # 각 막대를 오른쪽으로 5만큼 옮기고
        histogram.values(),  # 각 막대의 높이를 정해 주고
        10,  # 너비는 10으로 하자.
        edgecolor=(0, 0, 0))  # 각 막대의 테두리는 검은색으로 설정하자.

plt.axis([-5, 105, 0, 5])  # x축은 -5 부터 105, y축은 0부터 5

plt.xticks([10 * i for i in range(11)])  # x축의 레이블은 0, 10, ..., 100
plt.xlabel("Decile")  # 10분위 수
plt.ylabel("# of Students")

plt.title("Distribution of Exam 1 Grades")
plt.show()

"""
다음으로는 오해를 불러일으킬 수 있는 그래프를 살펴보자.
plt.axis 를 사용할 떄는 특히 신중해야 한다. 
막대 그래프를 그릴 때 y 축을 0에서 시작하지 않으면 다음과 같이 오해를 불러 일으키기 쉽기 때문이다.

2017년 과 2018년에 실제로는 data science 를 들은 횟수가 비슷함에도 불구하고,
단순 y 축 스케일의 잘못된 설정으로 큰 증가가 이뤄진 것으로 보여진다. 
"""
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("# of times I heard someone say data science")

# 오해를 불러일으키는 y축은 500 이상의 부분만 보여 줄 것이다.
# plt.axis([2016.5, 2018.5, 499, 506])
# plt.title("Look at the 'Huge' Increase!")
# plt.show()

# 올바르게 수정된 버전
plt.axis([2016.5, 2018.5, 0, 550])
plt.title("Not So Huge Anymore")
plt.show()
