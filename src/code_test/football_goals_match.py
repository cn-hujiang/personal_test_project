#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Time : 2023/2/6
# @Author : jiang.hu
# @File : football_goals_match.py
import random


class CalculationMatches(object):
    def __init__(self, teamA: list, teamB: list):
        self.teamA = teamA
        self.teamB = teamB
        self.result_match = []

    def calc(self):
        for goalsB in self.teamB:
            num = 0
            for goalsA in self.teamA:
                if not goalsA > goalsB:
                    num += 1
            self.result_match.append(num)


if __name__ == '__main__':
    # test
    team_a = [random.randint(0, 5) for i in range(10)]
    team_b = [random.randint(0, 5) for i in range(5)]
    # team_a = [1, 2, 3]
    # team_b = [2, 4]
    print(team_a)
    print(team_b)
    calc_match = CalculationMatches(team_a, team_b)
    calc_match.calc()
    print(calc_match.result_match)

