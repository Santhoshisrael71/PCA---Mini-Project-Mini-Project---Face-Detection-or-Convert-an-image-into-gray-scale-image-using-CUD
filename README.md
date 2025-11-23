# PCA-Mini-Project---Face-Detection-or-Convert-an-image-into-gray-scale-image-using-CUD
Mini Project - Face Detection or Convert an image into gray scale image using CUDA GPU programming

# AIM:

To design and implement a rain simulation system using both CPU (serial programming) and GPU (parallel CUDA programming) and to compare their performance based on execution time and frame rate, thereby demonstrating the advantage of GPU-based parallel computing in game programming.


# DEVELOPED BY :
```
NAME : SANTHOSH P
REG NO : 212224220088
```

# Required Software / Tools

C/C++ compiler

CUDA Toolkit (for GPU version)

Any simple text editor (VS Code / Notepad++)

Optional: OpenGL for visual animation

# PROCRDURE
1.Identify the problem
Rain simulation requires updating thousands of raindrops every frame, which is slow on a CPU.

2.Define raindrop properties
Each raindrop has position (x, y) and speed values.

3.Design CPU-based simulation
Use a loop to update each raindrop one by one (sequential processing).

4.Display rain output
Represent falling rain in the console or simple graphical window.

5.Analyze CPU performance
Measure frame rate (FPS) and execution time of the CPU loop.

6.Convert update logic to GPU kernel
Write a CUDA kernel where each thread handles one raindrop.

7.Allocate memory on GPU
Copy raindrop data (positions and speed) from CPU to GPU.

8.Run simulation on GPU
Launch the CUDA kernel with multiple blocks and threads (parallel processing).
9.
Measure GPU performance
Compare execution time and FPS against the CPU version.

10.Prepare comparison results
Show speed-up, graphs, and final conclusions describing how GPU performs faster.

# PROGRAM

RAIN PROGRAM IN CPU
```
%%writefile rain.cpp
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <unistd.h> // for usleep

using namespace std;

struct Drop {
    int x;
    int y;
    int speed;
};

const int width = 80;
const int height = 25;
const int numDrops = 200;

void clearScreen() {
    cout << "\033[2J\033[1;1H";  // ANSI escape sequence to clear screen (Linux)
}

int main() {
    srand(time(NULL));

    vector<Drop> rain(numDrops);

    for (int i = 0; i < numDrops; i++) {
        rain[i].x = rand() % width;
        rain[i].y = rand() % height;
        rain[i].speed = 1 + rand() % 3;
    }

    while (true) {
        clearScreen();

        char screen[height][width];
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                screen[i][j] = ' ';

        for (int i = 0; i < numDrops; i++) {
            rain[i].y += rain[i].speed;
            if (rain[i].y >= height) {
                rain[i].y = 0;
                rain[i].x = rand() % width;
            }
            screen[rain[i].y][rain[i].x] = '|';
        }

        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++)
                cout << screen[i][j];
            cout << "\n";
        }

        usleep(50000); // 50 ms
    }

    return 0;
}


!g++ rain.cpp -o rain
!./rain

```
GPU RUNTIME IN CPU RAIN PROGRAM

```
!pip install numba

import numpy as np
from numba import cuda
import math
import time

# GPU kernel
@cuda.jit
def update_rain(yPos, speed, height):
    idx = cuda.grid(1)
    if idx < yPos.size:
        yPos[idx] += speed[idx]
        if yPos[idx] >= height:
            yPos[idx] = 0

# Number of raindrops
n = 100000
height = 1080

# Initialize positions and speeds
yPos = np.random.randint(0, height, n).astype(np.float32)
speed = np.random.randint(1, 5, n).astype(np.float32)

# Copy to GPU
d_yPos = cuda.to_device(yPos)
d_speed = cuda.to_device(speed)

threads = 256
blocks = math.ceil(n / threads)

print("Running GPU simulation...")
start = time.time()

# Run 300 frames
for frame in range(300):
    update_rain[blocks, threads](d_yPos, d_speed, height)
cuda.synchronize()

end = time.time()

# Copy results back
result = d_yPos.copy_to_host()

print("Simulation Complete!")
print("Time taken:", end - start, "seconds")

# Show sample output
for i in range(10):
    print(f"Drop {i} yPos = {result[i]}")

```

# OUTPUT

CPU

<img width="832" height="526" alt="image" src="https://github.com/user-attachments/assets/876bc133-f52c-4cbc-b5d0-324c9ef1d71e" />

GPU RUNTIME 

<img width="1219" height="320" alt="image" src="https://github.com/user-attachments/assets/5754e186-f9a1-4ef8-a53a-76e3a4389051" />


# RESULT

The rain simulation mini-project was successfully implemented using both CPU and GPU approaches.
The CPU version updated each raindrop sequentially, while the GPU version processed all raindrops in parallel using CUDA kernel execution.
