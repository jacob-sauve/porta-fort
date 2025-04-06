# porta-fort
Simplified 2D Python snow particle physics simulation for use in a simulation for the Porta-Fort.  

## Install Dependencies and Initialise venv
To be able to run the [Taichi engine](https://pypi.org/project/taichi/) for the simulation, make sure that you have [Python 3.10](https://www.python.org/downloads/release/python-31016/) installed.  
Then, in the terminal, run:  
```
chmod +x setup.sh
./setup.sh
```
## Run the Simulation
To start the simulation, run:  
```
python3.10 main.py
```
With 8000 particles of radius 3 cm and an airbag expansion rate of 2 m/s, the following results are obtained:
![Animated simulation results with 8000 particles of radius 3 cm and airbag expansion rate of 2 m/s](/img/simulation_10K_3E-2_2.gif)
