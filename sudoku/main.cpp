#include "Sudoku.h"

int main(int argc, char **argv)
{
    Sudoku sudoku(5);
    sudoku.generateBoard();
    sudoku.solve();

    sudoku.printBoard();

    return 0;
}
