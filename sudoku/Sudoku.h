#ifndef SUDOKU_SUDOKU_H
#define SUDOKU_SUDOKU_H

#include <iostream>
#include <cstdio>
#include <algorithm>
#include <random>
#include <omp.h>

class Sudoku
{
public:
    Sudoku(int boardSize);
    ~Sudoku() {};

private:
    int getValue(int x, int y);
    void setValue(int x, int y, int value);
    void findEmptyItem(int &x, int &y);
    bool boardItemValid(int x0, int y0, int value);

public:
    bool solve();
    void generateBoard();
    void printBoard();

private:
    std::vector<int> board;
    int boardSize;
    int subBoardSize;
    int boardElementNum;

    // Random
    std::mt19937 mt;
    std::uniform_int_distribution<int> distr;
};


#endif //SUDOKU_SUDOKU_H
