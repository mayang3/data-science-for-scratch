from matplotlib import pyplot as plt

"""
pyplot 은 시각화를 단계별로 간편하게 만들 수 있는 구조로 되어 있으며, 
시각화가 완성되면 savefig() 를 통해 그래프를 저장하거나 show() 를 사용해서 화면에 띄울 수 있다.
"""

"""
====================================================================
                       2-1. 간단한 시각화
====================================================================
"""

years = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
gdp = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

# X 축에 연도, Y 축에 GDP 가 있는 선 그래프를 만든다.
plt.plot(years, gdp, color='green', marker='o', linestyle='solid')

# 제목 추가
plt.title("Nominal GDP")

# y 축에 레이블 추가
plt.ylabel("Billions of $")
# plt.savefig(fname="gdp graph.jpg")
plt.show()