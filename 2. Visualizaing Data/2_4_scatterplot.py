from matplotlib import pyplot as plt

"""
====================================================================
                       2-4. 산점도
====================================================================
"""

"""
산점도(scatterplot) 는 두 변수 간의 연관 관계를 보여주고 싶을 때 적합한 그래프이다.
예를 들어, 아래 예제는 각 사용자의 친구 수와 그들이 매일 사이트에서 체류하는 시간 사이의 연관성을 보여준다. 
"""

friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# 각각의 포인트에 레이블을 달아준다.
for label, friend_count, minute_count, in zip(labels, friends, minutes):
    plt.annotate(label,
                 xy=(friend_count, minute_count),  # 레이블을 데이터 포인트 근처에 두되
                 xytext=(5, -5),  # 약간 떨어져 있게 한다.
                 textcoords='offset points')

plt.title("Daily Minutes vs. Number of Friends")
plt.xlabel("# of friends")
plt.ylabel("daily minutes spent on the site")
plt.show()

# 변수들끼리 비교할때,
# matplotlib 이 자동으로 축의 범위를 설정하게 하면 그림 3-8 과 같이 공정한 비교를 하지 못하게 될 수 있다.

test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Axes Aren't Comparable")
plt.xlabel("test 1 grade")
plt.ylabel("test 2 grade")
# plt.axis("equal") # 이 명령을 추가하면 x 축과 y 축의 범위가 같아져서 공정한 비교를 할 수 있게 된다.
plt.show()
