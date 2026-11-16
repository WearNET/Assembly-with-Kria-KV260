# 2D convolution on the Kria KV260

In `project.ipynb`, you’ll take a photo with the AR1335 camera and use it to compare the same 2D convolution in Python, C, and AArch64 assembly. These implementations run on the KV260’s Arm processor. You’ll write one C function and two assembly routines, check their results, and use your measurements to explain which version ran fastest.

**At your station:** A prepared Kria KV260 with the AR1335 camera installed, its power cable, a connected Ethernet cable, and the IP address for your board’s Jupyter server. Jupyter is already running.

## 1. Work through the notebook

Open the supplied Jupyter address in your computer’s browser, including the port if one was provided. In the project folder, open **`project.ipynb`**.

Read and run the cells from top to bottom. Press **Shift + Enter** to run a cell and move to the next one. For **Q1–Q9**, double-click the **ANSWER Q# HERE** cell immediately below each question, enter your answer, and press **Shift + Enter**. Save your progress with **Ctrl + S**.

| Notebook part | What to do |
| --- | --- |
| 0 | Run the notebook’s readiness check. |
| 1 | Watch the convolution animation, compute one output pixel by hand, and run the photo cell to capture your input image with the AR1335. |
| 2 | Measure Python and C v1 on your image. |
| 3 | Write C v2 and compare how the loop order affects execution time. |
| 4 | Use the translation tables to connect the RISC-V instructions from Ripes to their AArch64 equivalents. |
| 5 | Write the inner loop of assembly v1. |
| 6 | Measure assembly v2, which uses pointers. |
| 7 | Write assembly v3 using the multiply-add instruction `madd`. |
| 8 | Measure assembly v4, which uses NEON vector instructions. |
| 9 | Run all versions together and record their execution times. |
| 10 | Watch the implementations race. |
| 10b | Run the live-video demonstration. Select a version to compare the camera image with its detected edges. Click **Stop video** when you finish so the camera is available again. |
| 11 | Use your measurements to explain the differences you observed. |

In **Parts 3, 5, and 7**, edit and run the exercise cell until it prints **`check: OK`**. If you see `NOT BUILT`, use the build message to locate the line to fix. If you see `MISMATCH`, the code ran but produced a different result. Check the loop bounds and image indices, or the operand order for `madd` in Part 7.

## 2. Hand in your work

1. Confirm that Q1–Q9 have answers, the three coding exercises show `check: OK`, and the Part 9 timing results and Part 11 conclusions are visible. The live video does not need to be saved.
2. Save **`project.ipynb`** with **Ctrl + S**. From Jupyter’s **File** menu, download the notebook in **`.ipynb`** format.
3. On your computer, rename the downloaded copy using your full name with no spaces, for example **`JuanPerezRivera.ipynb`**. Keep the project copy on the KV260 named `project.ipynb`.
4. Upload the renamed **`.ipynb`** file to **Moodle**. This notebook is the only file to submit.
