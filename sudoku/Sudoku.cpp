#include "Sudoku.h"

Sudoku::Sudoku(int size)
  : boardSize(size * size), subBoardSize(size), boardElementNum(boardSize * boardSize)
{
    std::random_device rd;
    mt = std::mt19937(rd());
    distr = std::uniform_int_distribution<int>(1, boardSize);
    board = std::vector<int>(boardElementNum, 0);
}

int Sudoku::getValue(int x, int y)
{
    return board[y * boardSize + x];
}

void Sudoku::setValue(int x, int y, int value)
{
    board[y * boardSize + x] = value;
}

void Sudoku::findEmptyItem(int &x, int &y)
{
    for (y = 0; y < boardSize; y++)
    {
        for (x = 0; x < boardSize; x++)
        {
            if (getValue(x, y) == 0)
                return;
        }
    }
}

bool Sudoku::boardItemValid(int x0, int y0, int value)
{
    for (int i = 0; i < boardSize; i++)
    {
        if ((i != x0) && (getValue(i, y0) == value))
            return false;
        if ((i != y0) && (getValue(x0, i) == value))
            return false;
    }

    int startX = x0 - x0 % subBoardSize;
    int startY = y0 - y0 % subBoardSize;
    for (int y = 0; y < subBoardSize; y++)
    {
        for (int x = 0; x < subBoardSize; x++)
        {
            if (((x != x0) || (y != y0)) && (getValue(startX + x, startY + y) == value))
                return false;
        }
    }

    return true;
}

bool Sudoku::solve()
{
    int emptyX = 0, emptyY = 0;
    findEmptyItem(emptyX, emptyY);
    if ((emptyX == boardSize) and (emptyY == boardSize))
        return true;

    for (int value = 1; value < boardSize + 1; value++)
    {
        if (boardItemValid(emptyX, emptyY, value))
        {
            setValue(emptyX, emptyY, value);
            if (solve())
                return true;
            setValue(emptyX, emptyY, 0);
        }
    }
    return false;
}

void Sudoku::generateBoard()
{
    for (;;)
    {
        std::fill(board.begin(), board.end(), 0);
        for (int y = 0; y < boardSize; y++)
        {
            for (int x = 0; x < boardSize; x++)
            {
                int value = distr(mt);
                if (value >= (int)(boardSize / 2))
                {
                    value = distr(mt);
                    setValue(x, y, value);
                    if (boardItemValid(x, y, value))
                        continue;
                    else
                        setValue(x, y, 0);
                }
            }
        }
        if (solve())
            return;
    }
}

void Sudoku::printBoard()
{
    for (int y = 0; y < boardSize; y++)
    {
        for (int x = 0; x < boardSize; x++)
        {
            printf("%2d ", getValue(x, y));
            if (((x + 1) % subBoardSize == 0) and (x != 0) and (x + 1 != boardSize))
                printf("| ");

            if (x == boardSize - 1)
                printf("\n");

            if ((x == boardSize - 1) and ((y + 1) % subBoardSize == 0) and (y + 1 != boardSize))
            {
                for (int i = 0; i < boardSize; i++)
                    printf("---");
                for (int i = 0; i < (2*subBoardSize - 3); i++)
                    printf("-");
                printf("\n");
            }
        }
    }
}
