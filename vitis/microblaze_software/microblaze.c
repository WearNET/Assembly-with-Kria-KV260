#include "xparameters.h"
#include "xgpio.h"
#include <stdint.h>

#define GPIO_DEVICE_ID XPAR_XGPIO_0_BASEADDR
#define CHANNEL1 1
#define CHANNEL2 2
#define IMAGE_WIDTH 1920
#define IMAGE_HEIGHT 1080
#define OUTPUT_WIDTH (IMAGE_WIDTH - 2)
#define OUTPUT_HEIGHT (IMAGE_HEIGHT - 2)


XGpio Gpio;

static int pixels[] = {
    #include "img_pixels.inc"
};

_Static_assert(sizeof pixels == IMAGE_WIDTH * IMAGE_HEIGHT, "Expected one grayscale byte per pixel");

static const int8_t edge_kernel[9] = {
    -1, -1, -1,
    -1,  8, -1,
    -1, -1, -1
};

static int edges[(IMAGE_HEIGHT - 2) * (IMAGE_WIDTH - 2)];

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
