import enum
import random


class Kid(enum.Enum):
    BOY = 0
    GIRL = 1


def random_kid() -> Kid:
    return random.choice([Kid.BOY, Kid.GIRL])


both_girl = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
    younger = random_kid()
    older = random_kid()
    # 첫째가 딸인 경우
    if older == Kid.GIRL:
        older_girl += 1
    # 둘 다 딸인 경우
    if older == Kid.GIRL and younger == Kid.GIRL:
        both_girl += 1
    # 둘 중에 하나만 딸인 경우 = 최소 하나가 딸인 경우
    if older == Kid.GIRL or younger == Kid.GIRL:
        either_girl += 1

print("P(both|older):", both_girl / older_girl)  # 0.5007089325501317 ~ 1/2
print("P(both|either):", both_girl / either_girl)  # 0.3311897106109325 ~ 1/3
