Open `src/main.cc` and change the parameters to match your in-game progress:

- `kExtractors` - set this to the list of extractors that you have unlocked
- `kBeltsPerExtractor` - set this to how many belts your extractors support

When done, run `./run.py release_main` (or `./run.py release_main.exe` - if using Windows).

The search tool finds all solutions with the minimum number of extractors + operations that have to be performed to obtain the result.

Result will look like this:

    1 = 1 [cost 0]
    2 = 2 [cost 0]
    3 = 3 [cost 0]
    4 = 4 [cost 0]
    5 = 5 [cost 0]
    6 = 6 [cost 0]
    7 = 7 [cost 0]
    8 = 8 [cost 0]
    9 = 9 [cost 0]
    10 = (5 + 5) [cost 2]
    11 = 11 [cost 0]
    12 = 12 [cost 0]
    13 = (7 + 6) [cost 3]
    13 = (8 + 5) [cost 3]
    13 = (9 + 4) [cost 3]
    13 = (11 + 2) [cost 3]
    13 = (12 + 1) [cost 3]
    14 = (7 + 7) [cost 2]
    15 = (5 * 3) [cost 3]
    15 = (8 + 7) [cost 3]
    15 = (9 + 6) [cost 3]
    15 = (11 + 4) [cost 3]
    15 = (12 + 3) [cost 3]
    15 = ((5 + 5) + 5) [cost 3]
    16 = (4 * 4) [cost 2]
    16 = (8 + 8) [cost 2]
    17 = (9 + 8) [cost 3]
    17 = (11 + 6) [cost 3]
    17 = (12 + 5) [cost 3]
    18 = (9 + 9) [cost 2]
    19 = (11 + 8) [cost 3]
    19 = (12 + 7) [cost 3]
    20 = (5 * 4) [cost 3]
    20 = (11 + 9) [cost 3]
    20 = (12 + 8) [cost 3]
    20 = ((4 * 4) + 4) [cost 3]
    20 = ((5 * 5) - 5) [cost 3]
    21 = (7 * 3) [cost 3]
    21 = (12 + 9) [cost 3]
    21 = ((7 + 7) + 7) [cost 3]
    22 = (11 + 11) [cost 2]
    23 = (12 + 11) [cost 3]
    24 = (12 + 12) [cost 2]
    25 = (5 * 5) [cost 2]
    26 = ((5 * 5) + 1) [cost 4]
    26 = ((7 + 7) + 12) [cost 4]
    26 = ((9 + 8) + 9) [cost 4]
    26 = ((11 + 2) * 2) [cost 4]
    26 = ((11 + 4) + 11) [cost 4]
    26 = ((12 * 2) + 2) [cost 4]
    26 = ((12 + 12) + 2) [cost 4]
    27 = (9 * 3) [cost 3]
    27 = ((9 + 9) + 9) [cost 3]
    28 = (7 * 4) [cost 3]
    29 = ((5 * 5) + 4) [cost 4]
    29 = ((6 * 6) - 7) [cost 4]
    29 = ((9 + 9) + 11) [cost 4]
    29 = ((11 + 11) + 7) [cost 4]
    29 = ((12 + 5) + 12) [cost 4]
    30 = (6 * 5) [cost 3]
    30 = ((5 * 5) + 5) [cost 3]
    30 = ((6 * 6) - 6) [cost 3]
    31 = ((5 * 5) + 6) [cost 4]
    31 = ((6 * 6) - 5) [cost 4]
    31 = ((11 + 9) + 11) [cost 4]
    31 = ((12 + 7) + 12) [cost 4]
    32 = (8 * 4) [cost 3]
    33 = (11 * 3) [cost 3]
    33 = ((11 + 11) + 11) [cost 3]
    34 = ((5 * 5) + 9) [cost 4]
    34 = ((6 * 6) - 2) [cost 4]
    34 = ((11 + 11) + 12) [cost 4]
    35 = (7 * 5) [cost 3]
    36 = (6 * 6) [cost 2]
    37 = ((5 * 5) + 12) [cost 4]
    37 = ((6 * 6) + 1) [cost 4]
    37 = ((7 * 7) - 12) [cost 4]
    38 = ((6 * 6) + 2) [cost 4]
    38 = ((7 * 7) - 11) [cost 4]
    39 = ((6 * 6) + 3) [cost 4]
    40 = (8 * 5) [cost 3]
    41 = ((6 * 6) + 5) [cost 4]
    41 = ((7 * 7) - 8) [cost 4]
    42 = (7 * 6) [cost 3]
    42 = ((6 * 6) + 6) [cost 3]
    42 = ((7 * 7) - 7) [cost 3]
    43 = ((6 * 6) + 7) [cost 4]
    43 = ((7 * 7) - 6) [cost 4]
    44 = (11 * 4) [cost 3]
    45 = (9 * 5) [cost 3]
    46 = ((7 * 7) - 3) [cost 4]
    47 = ((6 * 6) + 11) [cost 4]
    47 = ((7 * 7) - 2) [cost 4]
    48 = (8 * 6) [cost 3]
    48 = (12 * 4) [cost 3]
    49 = (7 * 7) [cost 2]
    50 = ((5 + 5) * 5) [cost 3]
    51 = ((7 * 7) + 2) [cost 4]
    52 = ((7 * 7) + 3) [cost 4]
    52 = ((8 * 8) - 12) [cost 4]
    52 = ((9 + 4) * 4) [cost 4]
    52 = ((12 * 4) + 4) [cost 4]
    53 = ((7 * 7) + 4) [cost 4]
    53 = ((8 * 8) - 11) [cost 4]
    54 = (9 * 6) [cost 3]
    55 = (11 * 5) [cost 3]
    56 = (8 * 7) [cost 3]
    56 = ((7 * 7) + 7) [cost 3]
    56 = ((8 * 8) - 8) [cost 3]
    57 = ((7 * 7) + 8) [cost 4]
    57 = ((8 * 8) - 7) [cost 4]
    58 = ((7 * 7) + 9) [cost 4]
    58 = ((8 * 8) - 6) [cost 4]
    59 = ((8 * 8) - 5) [cost 4]
    60 = (12 * 5) [cost 3]
    61 = ((7 * 7) + 12) [cost 4]
    61 = ((8 * 8) - 3) [cost 4]
    62 = ((8 * 8) - 2) [cost 4]
    63 = (9 * 7) [cost 3]
    64 = (8 * 8) [cost 2]
    65 = ((8 + 5) * 5) [cost 4]
    65 = ((8 * 8) + 1) [cost 4]
    65 = ((12 * 5) + 5) [cost 4]
    66 = (11 * 6) [cost 3]
    67 = ((8 * 8) + 3) [cost 4]
    68 = ((8 * 8) + 4) [cost 4]
    69 = ((8 * 8) + 5) [cost 4]
    69 = ((9 * 9) - 12) [cost 4]
    70 = ((5 + 5) * 7) [cost 4]
    70 = ((7 + 7) * 5) [cost 4]
    70 = ((8 * 8) + 6) [cost 4]
    70 = ((9 * 7) + 7) [cost 4]
    70 = ((9 * 9) - 11) [cost 4]
    70 = ((11 * 7) - 7) [cost 4]
    71 = ((8 * 8) + 7) [cost 4]
    72 = (9 * 8) [cost 3]
    72 = (12 * 6) [cost 3]
    72 = ((8 * 8) + 8) [cost 3]
    72 = ((9 * 9) - 9) [cost 3]
    73 = ((8 * 8) + 9) [cost 4]
    73 = ((9 * 9) - 8) [cost 4]
    74 = ((9 * 9) - 7) [cost 4]
    75 = ((5 * 3) * 5) [cost 4]
    75 = ((8 * 8) + 11) [cost 4]
    75 = ((9 * 9) - 6) [cost 4]
    75 = (((5 + 5) + 5) * 5) [cost 4]
    76 = ((8 * 8) + 12) [cost 4]
    76 = ((9 * 9) - 5) [cost 4]
    77 = (11 * 7) [cost 3]
    78 = ((7 + 6) * 6) [cost 4]
    78 = ((9 * 9) - 3) [cost 4]
    78 = ((12 * 6) + 6) [cost 4]
    79 = ((9 * 9) - 2) [cost 4]
    80 = ((4 * 4) * 5) [cost 4]
    80 = ((5 + 5) * 8) [cost 4]
    80 = ((8 + 8) * 5) [cost 4]
    80 = ((8 * 8) + (8 + 8)) [cost 4]
    80 = ((9 * 8) + 8) [cost 4]
    80 = ((9 * 9) - 1) [cost 4]
    80 = ((11 * 8) - 8) [cost 4]
    80 = (((4 * 4) + 4) * 4) [cost 4]
    81 = (9 * 9) [cost 2]
    82 = ((9 * 9) + 1) [cost 4]
    83 = ((9 * 9) + 2) [cost 4]
    84 = (12 * 7) [cost 3]
    85 = ((9 * 9) + 4) [cost 4]
    85 = ((12 + 5) * 5) [cost 4]
    86 = ((9 * 9) + 5) [cost 4]
    87 = ((9 * 9) + 6) [cost 4]
    88 = (11 * 8) [cost 3]
    89 = ((9 * 9) + 8) [cost 4]
    90 = ((9 * 9) + 9) [cost 3]
    91 = ((7 + 6) * 7) [cost 4]
    91 = ((12 * 7) + 7) [cost 4]
    91 = (((7 + 7) * 7) - 7) [cost 4]
    92 = ((9 * 9) + 11) [cost 4]
    93 = ((9 * 9) + 12) [cost 4]
    94 = ((9 * 9) + (9 + 4)) [cost 5]
    94 = ((11 * 8) + 6) [cost 5]
    94 = ((11 * 9) - 5) [cost 5]
    94 = ((12 * 8) - 2) [cost 5]
    94 = (((7 + 7) * 7) - 4) [cost 5]
    94 = (((7 * 7) - 2) * 2) [cost 5]
    95 = ((9 * 9) + (7 + 7)) [cost 5]
    95 = ((11 + 8) * 5) [cost 5]
    95 = ((11 * 8) + 7) [cost 5]
    95 = ((11 * 9) - 4) [cost 5]
    95 = ((12 + 7) * 5) [cost 5]
    95 = ((12 * 7) + 11) [cost 5]
    95 = ((12 * 8) - 1) [cost 5]
    95 = ((12 * 12) - (7 * 7)) [cost 5]
    95 = (((5 * 4) * 5) - 5) [cost 5]
    95 = (((5 + 5) * 9) + 5) [cost 5]
    95 = (((7 + 7) * 7) - 3) [cost 5]
    95 = (((9 + 9) * 5) + 5) [cost 5]
    95 = (((9 * 9) + 5) + 9) [cost 5]
    96 = (12 * 8) [cost 3]
    97 = ((9 * 9) + (4 * 4)) [cost 5]
    97 = ((9 * 9) + (8 + 8)) [cost 5]
    97 = ((11 * 8) + 9) [cost 5]
    97 = ((11 * 9) - 2) [cost 5]
    97 = ((12 * 8) + 1) [cost 5]
    97 = ((12 * 9) - 11) [cost 5]
    97 = ((11 * 11) - (12 + 12)) [cost 5]
    97 = (((7 + 6) * 7) + 6) [cost 5]
    97 = (((7 + 7) * 7) - 1) [cost 5]
    97 = (((8 + 7) * 7) - 8) [cost 5]
    97 = (((9 * 9) + 9) + 7) [cost 5]
    97 = (((12 + 5) * 5) + 12) [cost 5]
    98 = ((7 + 7) * 7) [cost 3]
    99 = (11 * 9) [cost 3]
    100 = ((5 * 4) * 5) [cost 4]
    100 = ((5 + 5) * (5 + 5)) [cost 4]
