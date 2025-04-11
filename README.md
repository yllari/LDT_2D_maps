# LDT_2D_maps

This is a repository containing the following:
├── DOOB
│   ├── doob_calc.py
│   ├── doob.py
│   └── func_plot.py
├── README.md
├── requirements.txt
├── running_indicator
│   ├── plot.py
│   └── running_indicator.py
└── SCGF
    ├── scgf.py
    └── simple_plot.py

DOOB: Program that generates the conjugate map via Doob transformation.
SCGF: Parallelized SCGF calculation with multiprocessing

The standard procedure to run the code would be to follow the instructions inside DOOB/doob.py, that is, 
modify the constants and functions at the beginning of the code and run it

Additionally, there is:

running_indicator: an application of an indicator function to
find the periodic orbits in a given map
