#include "xparameters.h"
#include "xgpio.h"

#define GPIO_DEVICE_ID XPAR_XGPIO_0_BASEADDR
#define CHANNEL1 1
#define CHANNEL2 2

extern void BruteForce(void);
extern void Flattened_Loops(void);
extern void Serial_Traversal(void);
extern void Strengh_Reduction(void);

XGpio Gpio;

int main() {
    int status;

    status = XGpio_Initialize(&Gpio, GPIO_DEVICE_ID);
    if (status != XST_SUCCESS){
        return XST_FAILURE;
    }

    XGpio_SetDataDirection(&Gpio, CHANNEL1, 0xFFFFFFFF); // input
    XGpio_SetDataDirection(&Gpio, CHANNEL2, 0x00000000); // output

    while (1) {
        int start = XGpio_DiscreteRead(&Gpio, CHANNEL1);
        if (start & 1) {
            XGpio_DiscreteWrite(&Gpio, CHANNEL2, 0x0); // example

            //ADD YOUR CODE HERE

            XGpio_DiscreteWrite(&Gpio, CHANNEL2, 0x80000064); // DONE
        }
    }

    return 0;
}
